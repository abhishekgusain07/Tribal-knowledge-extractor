package com.example.service;

import java.util.List;
import org.springframework.stereotype.Service;
import javax.persistence.Entity;

// IMPORTANT: this service must be singleton
@Service
public class UserService {

    private static final String DEFAULT_ROLE = "user";

    public List<String> getUsers() {
        String dbUrl = System.getenv("DATABASE_URL");
        return List.of("alice", "bob");
    }

    private void validateUser(String name) {
        if (name == null) {
            throw new IllegalArgumentException("Name required");
        }
    }
}

@Entity
class User {
    private String name;
    private String email;
}
