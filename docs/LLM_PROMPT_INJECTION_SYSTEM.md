# LLM Prompt Injection System

This system automatically injects prompts from the `docs/prompts/` folder into both Mistral and GPT LLM services, providing consistent behavior and specialized instructions.

## 🎯 Overview

The prompt injection system allows you to:
- Store specialized prompts in text files
- Automatically load and inject them into LLM responses
- Manage prompts dynamically without service restarts
- Apply consistent behavior across all LLM services

## 📁 File Structure

```
docs/prompts/
├── prompt-main.txt          # Main banking/loan specialist prompt
└── [additional-prompts].txt # Add more prompts as needed
```

## 🚀 How It Works

1. **Automatic Loading**: Both Mistral and GPT LLM services automatically load all `.txt` files from `docs/prompts/`
2. **System Prompt Generation**: All prompts are combined into a comprehensive system prompt
3. **Request Enhancement**: Every user request gets the system prompt prepended as context
4. **Dynamic Updates**: Prompts can be reloaded without restarting services

## 📝 Current Prompts

### prompt-main.txt
- **Purpose**: Banking loan specialist persona (Alex from Finally Payoff Debt)
- **Content**: Complete conversation flow for debt consolidation calls
- **Features**: 
  - Qualifying questions and requirements
  - Objection handling scripts
  - Compliance and legal guidelines
  - Personality and tone instructions

## 🔧 Management Tools

### Prompt Manager Script
```bash
# List all available prompts
python scripts/utilities/prompt_manager.py list

# Show content of a specific prompt
python scripts/utilities/prompt_manager.py show prompt-main

# Test prompt loading system
python scripts/utilities/prompt_manager.py test

# Check and reload prompts in running LLM services
python scripts/utilities/prompt_manager.py services

# Reload prompts from disk
python scripts/utilities/prompt_manager.py reload
```

### Test Script
```bash
# Test prompt injection in both LLM services
python scripts/utilities/test_prompt_injection.py
```

## 🌐 API Endpoints

Both Mistral (port 8021) and GPT (port 8022) services provide:

### Get Prompts Information
```bash
GET http://localhost:8021/prompts
GET http://localhost:8022/prompts
```
Returns information about loaded prompts, system prompt length, and preview.

### Reload Prompts
```bash
POST http://localhost:8021/prompts/reload
POST http://localhost:8022/prompts/reload
```
Reloads all prompts from the `docs/prompts/` folder without restarting.

### Get Specific Prompt
```bash
GET http://localhost:8021/prompts/prompt-main
GET http://localhost:8022/prompts/prompt-main
```
Returns the content of a specific prompt file.

## 📋 Usage Examples

### Adding New Prompts
1. Create a new `.txt` file in `docs/prompts/`
2. Add your prompt content
3. Reload prompts: `POST /prompts/reload`
4. The new prompt will be automatically included in system prompts

### Testing Prompt Injection
```python
import requests

# Test with Mistral LLM
response = requests.post("http://localhost:8021/generate", json={
    "text": "I need help with debt consolidation",
    "use_cache": False
})

# Response should include elements from the prompt (Alex, Finally Payoff Debt, etc.)
print(response.json()["response"])
```

## 🎭 Prompt Format

Each prompt file should contain:
- Clear instructions for the AI persona
- Specific conversation flows or responses
- Compliance and legal requirements
- Tone and personality guidelines

Example structure:
```
=== ROLE DEFINITION ===
Your name is Alex and you are a prequalification specialist...

=== CONVERSATION FLOW ===
1. Greeting and introduction
2. Qualifying questions
3. Objection handling
...

=== COMPLIANCE ===
Legal requirements and restrictions...
```

## 🔄 System Integration

The prompt injection system integrates seamlessly with:
- **Service Combinations**: All LLM services in combinations use prompts
- **Caching**: Prompts are cached for performance
- **Logging**: Prompt injection events are logged
- **Error Handling**: Graceful fallback if prompts fail to load

## 💡 Best Practices

1. **File Naming**: Use descriptive names (e.g., `banking-specialist.txt`, `customer-service.txt`)
2. **Content Length**: Keep prompts focused and concise for optimal performance
3. **Regular Updates**: Use the reload endpoint to update prompts without downtime
4. **Testing**: Always test prompt changes with the test script
5. **Version Control**: Track prompt changes in your repository

## 🚨 Important Notes

- Prompts are automatically loaded on service startup
- System prompt is prepended to every user request
- Empty or invalid prompt files are skipped
- Prompt injection is logged for monitoring
- All LLM services share the same prompt library

## 🔍 Troubleshooting

### Prompts Not Loading
1. Check file permissions on `docs/prompts/` folder
2. Ensure `.txt` file extension
3. Verify file encoding is UTF-8
4. Check service logs for error messages

### Prompts Not Applied
1. Verify services are using the latest prompts: `GET /prompts`
2. Reload prompts: `POST /prompts/reload`
3. Check system prompt length in API response
4. Monitor logs for prompt injection messages

### Performance Issues
1. Keep individual prompt files under 10KB
2. Limit total number of prompt files
3. Monitor system prompt length (recommended < 50KB)
4. Use caching effectively
