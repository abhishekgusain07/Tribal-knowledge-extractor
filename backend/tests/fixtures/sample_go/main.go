package main

import (
	"fmt"
	"net/http"
	"os"
)

// HACK: this timeout is too low for production
const DefaultTimeout = 30

type Server struct {
	Host string
	Port int
}

func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%d", s.Host, s.Port)
	return http.ListenAndServe(addr, nil)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	apiKey := os.Getenv("API_KEY")
	if apiKey == "" {
		w.WriteHeader(500)
		return
	}
	w.WriteHeader(200)
}

func main() {
	http.HandleFunc("/health", handleHealth)
	s := &Server{Host: "localhost", Port: 8080}
	s.Start()
}
