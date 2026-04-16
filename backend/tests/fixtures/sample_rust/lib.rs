use std::collections::HashMap;
use std::env;

// TODO: add proper error handling
const MAX_RETRIES: u32 = 3;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
}

pub trait Handler {
    fn handle(&self, request: &str) -> String;
}

impl Config {
    pub fn from_env() -> Self {
        let host = env::var("HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = env::var("PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse()
            .unwrap();
        Config { host, port }
    }
}

impl Handler for Config {
    fn handle(&self, request: &str) -> String {
        format!("Handled: {}", request)
    }
}

fn internal_helper() -> HashMap<String, String> {
    HashMap::new()
}
