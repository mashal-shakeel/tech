## Day 2 

### 1. Pydantic Schema Validation
- Added Pydantic for schema validation
- reject invalid JSON responses
- used `extra="forbid"` to prevent unexpected fields.

### 2. JSON Extraction 
- extraced JSON from model responses before validation.

### 3. Confidence Scoring
- Added a `confidence` field (0-1) where model rates confidence in its response.

### 4. Human Review Flag
- i added `needs_review` boolean field which is set to `True` when confidence is below 0.7

### 5. Retry Logic
- added automatic retry mechanism with 3 max tries
- confidence < 0.7 responses trigger re-evaluation

### 6. Fallback Strategy
- tracks the highest-confidence valid response across all attempts.
- but if no response reaches the confidence threshold 0.7, code returns the best available result out of the three

### 7. Streaming Responses
- Added streaming for email generation.

### 8. Prompt Injection 
- Updated system prompts with instructions to treat user input as data, not instructions.

### 9. Hallucination 
- Added instructions to avoid guessing or fabricating information.

### 10. Cost Optimization
- Implemented early-return logic
