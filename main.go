package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/mux"
)

// Student represents the student model
type Student struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Age   int    `json:"age"`
	Email string `json:"email"`
}

// StudentStore handles the in-memory storage of students with thread-safety
type StudentStore struct {
	sync.RWMutex
	students map[int]Student
	nextID   int
}

// NewStudentStore creates a new instance of StudentStore
func NewStudentStore() *StudentStore {
	return &StudentStore{
		students: make(map[int]Student),
		nextID:   1,
	}
}

// ValidationError represents an error during validation
type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
}

// OllamaRequest represents the request structure for Ollama API
type OllamaRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

// OllamaResponse represents the response structure from Ollama API
type OllamaResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

// Validate checks if the student data is valid
func (s *Student) Validate() []ValidationError {
	var errors []ValidationError

	if s.Name == "" {
		errors = append(errors, ValidationError{Field: "name", Message: "Name is required"})
	}

	if s.Age < 0 || s.Age > 150 {
		errors = append(errors, ValidationError{Field: "age", Message: "Age must be between 0 and 150"})
	}

	if s.Email == "" {
		errors = append(errors, ValidationError{Field: "email", Message: "Email is required"})
	}

	return errors
}

// CreateStudent handles the creation of a new student
func (store *StudentStore) CreateStudent(w http.ResponseWriter, r *http.Request) {
	var student Student
	if err := json.NewDecoder(r.Body).Decode(&student); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if errors := student.Validate(); len(errors) > 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(errors)
		return
	}

	store.Lock()
	student.ID = store.nextID
	store.students[student.ID] = student
	store.nextID++
	store.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(student)
}

// GetAllStudents returns all students
func (store *StudentStore) GetAllStudents(w http.ResponseWriter, r *http.Request) {
	store.RLock()
	students := make([]Student, 0, len(store.students))
	for _, student := range store.students {
		students = append(students, student)
	}
	store.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(students)
}

// GetStudent returns a specific student by ID
func (store *StudentStore) GetStudent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := 0
	fmt.Sscanf(vars["id"], "%d", &id)

	store.RLock()
	student, exists := store.students[id]
	store.RUnlock()

	if !exists {
		http.Error(w, "Student not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(student)
}

// UpdateStudent updates a student by ID
func (store *StudentStore) UpdateStudent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := 0
	fmt.Sscanf(vars["id"], "%d", &id)

	var student Student
	if err := json.NewDecoder(r.Body).Decode(&student); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if errors := student.Validate(); len(errors) > 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(errors)
		return
	}

	store.Lock()
	if _, exists := store.students[id]; !exists {
		store.Unlock()
		http.Error(w, "Student not found", http.StatusNotFound)
		return
	}

	student.ID = id
	store.students[id] = student
	store.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(student)
}

// DeleteStudent deletes a student by ID
func (store *StudentStore) DeleteStudent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := 0
	fmt.Sscanf(vars["id"], "%d", &id)

	store.Lock()
	if _, exists := store.students[id]; !exists {
		store.Unlock()
		http.Error(w, "Student not found", http.StatusNotFound)
		return
	}

	delete(store.students, id)
	store.Unlock()

	w.WriteHeader(http.StatusNoContent)
}

// GetStudentSummary generates an AI-based summary of a student using Ollama
func (store *StudentStore) GetStudentSummary(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := 0
	fmt.Sscanf(vars["id"], "%d", &id)

	store.RLock()
	student, exists := store.students[id]
	store.RUnlock()

	if !exists {
		http.Error(w, "Student not found", http.StatusNotFound)
		return
	}

	// Prepare the prompt for Ollama
	prompt := fmt.Sprintf(
		"Generate a brief summary of this student:\nName: %s\nAge: %d\nEmail: %s\n"+
			"Focus on their basic information and potential academic journey based on their age.",
		student.Name, student.Age, student.Email,
	)

	// Create Ollama request
	ollamaReq := OllamaRequest{
		Model: "llama2",
		Messages: []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	// Convert request to JSON
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		http.Error(w, "Failed to create Ollama request", http.StatusInternalServerError)
		return
	}

	// Make request to Ollama API
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		http.Error(w, "Failed to connect to Ollama API", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	// Parse Ollama response
	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		http.Error(w, "Failed to parse Ollama response", http.StatusInternalServerError)
		return
	}

	// Return the summary
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"summary": ollamaResp.Message.Content,
	})
}

func main() {
	store := NewStudentStore()
	router := mux.NewRouter()

	// Register routes
	router.HandleFunc("/students", store.CreateStudent).Methods("POST")
	router.HandleFunc("/students", store.GetAllStudents).Methods("GET")
	router.HandleFunc("/students/{id}", store.GetStudent).Methods("GET")
	router.HandleFunc("/students/{id}", store.UpdateStudent).Methods("PUT")
	router.HandleFunc("/students/{id}", store.DeleteStudent).Methods("DELETE")
	router.HandleFunc("/students/{id}/summary", store.GetStudentSummary).Methods("GET")

	// Start server
	log.Printf("Server starting on :8080")
	if err := http.ListenAndServe(":8080", router); err != nil {
		log.Fatal(err)
	}
}