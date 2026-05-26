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
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());


    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key.clone());

    let client = Client::with_config(config);

    let mut history : Vec<Value> = vec![json!({
        "role": "user",
        "content": &args.prompt
    })];

    loop {
        // api call
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
                }],
                "model": "anthropic/claude-haiku-4.5",
            }))
            .await?;

        eprintln!("Logs from your program will appear here!");

        let msg = response["choices"][0]["message"].clone();
        history.push(msg.clone());
        // Iterate over tool calls
        if let Some(tool_calls) = msg["tool_calls"].as_array() {
            for tool_call in tool_calls {
                // Identify the tool call and its arguments
                if let Some(function) = tool_call["function"].as_object() {
                    // Parse the tool call name and arguments
                    if let Some(arguments) = function["arguments"].as_str() {
                        let parsed: serde_json::Value = serde_json::from_str(arguments)?;
                        
                        if let Some(file_path) = parsed["file_path"].as_str() {
                            let data = fs::read_to_string(file_path).await?;
                            history.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": data
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
