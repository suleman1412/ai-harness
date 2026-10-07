use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use tokio::fs;
use std::{env, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());


    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let model = env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openrouter/free".to_string());

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let mut history : Vec<Value> = vec![json!({
        "role": "user",
        "content": &args.prompt
    })];

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": history,
                "tools": [{
                  "type": "function",
                  "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "file_path": {
                          "type": "string",
                          "description": "The path to the file to read"
                        }
                      },
                      "required": ["file_path"]
                    }
                  }
                }, 
                {
                  "type": "function",
                  "function": {
                    "name": "Write",
                    "description": "Write content to a file",
                    "parameters": {
                      "type": "object",
                      "required": ["file_path", "content"],
                      "properties": {
                        "file_path": {
                          "type": "string",
                          "description": "The path of the file to write to"
                        },
                        "content": {
                          "type": "string",
                          "description": "The content to write to the file"
                        }
                      }
                    }
                  }
                },
                {
                  "type": "function",
                  "function": {
                    "name": "Bash",
                    "description": "Execute a shell command",
                    "parameters": {
                      "type": "object",
                      "required": ["command"],
                      "properties": {
                        "command": {
                          "type": "string",
                          "description": "The command to execute"
                        }
                      }
                    }
                  }
                }],
                "model": model,
            }))
            .await?;
        let msg = response["choices"][0]["message"].clone();
        history.push(msg.clone());
        // Iterate over tool calls
        if let Some(tool_calls) = msg["tool_calls"].as_array() {
            for tool_call in tool_calls {
                // Identify the tool call and its arguments
                if let Some(function) = tool_call["function"].as_object() {
                    // Parse the tool call name and arguments
                    if let Some("Read") = function["name"].as_str() {
                        if let Some(arguments) = function["arguments"].as_str() {
                            if let Ok(data) = read_tool_call(arguments).await{
                                history.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": data
                                }));
                            }
                        }
                    } else if let Some("Write") = function["name"].as_str() {
                        if let Some(arguments) = function["arguments"].as_str() {
                            if let Ok(_) = write_tool_call(arguments).await{
                                history.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": "File written successfully"
                                }));
                            }
                        }
                    } else if let Some("Bash") = function["name"].as_str() {
                        if let Some(arguments) = function["arguments"].as_str() {
                            let result = bash_tool_call(arguments).await;
                            let content = match result {
                                Ok(out) => out,
                                Err(e) => format!("Error: {}", e),
                            };
                            history.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": content
                            }));
                        }
                    }
                }
            }
        }
        else if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
            println!("{}", content);
            break;
        }
    }

    Ok(())
}


async fn read_tool_call(arguments: &str) -> Result<String, Box<dyn std::error::Error>> {
    let parsed_args: serde_json::Value = serde_json::from_str(arguments)?;
    if let Some(file_path) = parsed_args["file_path"].as_str() {
        let data = fs::read_to_string(file_path).await?;
        Ok(data)
    } else {
        Err("file_path doesnt exist".into())
    }
}
async fn write_tool_call(arguments: &str) -> Result<(), Box<dyn std::error::Error>> {
    let parsed_args: serde_json::Value = serde_json::from_str(arguments)?;
    let content_of_file = parsed_args["content"].as_str().unwrap();
    let file_path = parsed_args["file_path"].as_str().unwrap();

    if let Err(e) = read_tool_call(arguments).await {
           eprintln!("Read_tool_call failed: {e}");
    }
    let data = fs::write(file_path, content_of_file).await?;
    Ok(data)
}

async fn bash_tool_call(arguments: &str) -> Result<String, Box<dyn std::error::Error>> {
    let parsed_args: serde_json::Value = serde_json::from_str(arguments)?;
    if let Some(command) = parsed_args["command"].as_str() {
        let output = std::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()?;
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(format!("Command failed: {stderr}").into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err("command not found".into())
    }
}
