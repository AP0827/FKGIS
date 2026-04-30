# Web Crawl Agent - System Methodology

## Overview

The Web Crawl Agent is an intelligent content analysis system that can summarize websites, raw text, and documents using AI-powered analysis. It adapts its summary structure based on content type, providing context-aware insights.

## Architecture

### 1. **Input Processing Layer**

The system supports three input modes:

#### A. Website Crawling
- **Technology**: Playwright (headless Chromium browser)
- **Process**:
  1. Launches headless browser session
  2. Navigates to target URL
  3. Extracts text content, links, headings, and metadata
  4. Follows internal links up to configured depth (`crawl_max_pages`)
  5. Aggregates content across multiple pages
  6. Builds analysis summary with metrics (links, keywords, CTAs)

#### B. Text Input
- **Process**:
  1. Accepts raw text directly from user
  2. No web crawling required
  3. Sends text directly to LLM for analysis

#### C. Document Upload
- **Supported Formats**: `.txt`, `.md`, `.pdf`
- **Process**:
  1. Extracts text from uploaded file
  2. PDF extraction uses PyPDF2 library
  3. Text files decoded with UTF-8/Latin-1 fallback
  4. Extracted text sent to LLM for analysis

### 2. **Content Analysis Layer**

#### A. Content Type Detection
The system uses intelligent prompting to detect content type:
- **Conversations/Dialogues**: Identifies speakers, topics, decisions
- **Articles/Documents**: Extracts themes, arguments, conclusions
- **Meeting Notes**: Finds participants, agenda items, action items
- **General Content**: Adapts to content structure

#### B. LLM Integration

**Supported Providers**:
1. **Google Gemini** (default)
   - Model: `gemini-2.5-flash`
   - API: REST via `generativelanguage.googleapis.com`
   - Structured output: JSON format

2. **xAI Grok**
   - Model: `grok-2-latest`
   - API: REST via `api.x.ai`
   - Structured output: JSON schema

**Prompt Engineering Strategy**:
- **Context-Aware Prompts**: Different prompts for different content types
- **Structured Output**: Requests JSON with specific schema
- **Dynamic Sections**: Allows LLM to create relevant sections based on content
- **Fallback Handling**: Graceful degradation if LLM blocks content

### 3. **Summary Generation Methodology**

#### A. Dynamic Schema Approach

**Problem**: Fixed schemas don't work for diverse content types
- Conversations need: speakers, topics_discussed, decisions
- Articles need: main_themes, key_arguments, conclusions
- Meetings need: participants, agenda_items, action_items

**Solution**: Flexible, context-aware schema

```json
{
  "overview": "2-4 sentence comprehensive synopsis",
  "content_type": "detected type (conversation/article/meeting/etc.)",
  "sections": {
    "dynamic_section_name": ["item1", "item2", ...],
    ...
  }
}
```

#### B. Prompt Design

**For Text/Documents**:
1. Instructs LLM to detect content type first
2. Provides examples of appropriate sections for each type
3. Requests 3-6 relevant sections
4. Allows LLM creativity in section naming

**For Websites**:
1. Includes crawl metadata (keywords, links, CTAs)
2. Blends structured data with extracted text
3. Focuses on website-specific insights

#### C. Schema Limitations & Workarounds

**Gemini API Limitation**:
- Doesn't support `additionalProperties` in responseSchema
- Can't enforce dynamic object structures via schema

**Workaround**:
- For text/document: Remove schema validation, rely on prompt instructions
- For websites: Use fixed schema (key_sections, highlights, recommendations)
- Parse JSON manually with error handling

### 4. **Report Generation**

#### A. PDF Report Builder
- **Library**: fpdf2
- **Structure**:
  1. Header with title and metadata
  2. Overview section (emphasized)
  3. Dynamic sections based on content type
  4. Metrics (only for website crawls)

#### B. Section Rendering
- Maps technical section names to human-readable titles
- Only renders non-empty sections
- Handles both new dynamic format and legacy format

#### C. Frontend Display
- Real-time progress updates via Server-Sent Events (SSE)
- Dynamic section rendering based on content type
- Responsive UI with mode switching (URL/Text/Document)

### 5. **Error Handling & Resilience**

#### A. LLM Error Handling
- **Content Blocking**: Falls back to basic summary
- **API Errors**: Shows detailed error messages
- **Network Errors**: Graceful timeout handling

#### B. Content Extraction Errors
- PDF extraction failures: Clear error messages
- Encoding issues: Multiple encoding attempts (UTF-8 → Latin-1)
- Empty content: Validation before processing

#### C. User Experience
- Progress indicators during processing
- Clear error messages with actionable suggestions
- Fallback summaries when LLM unavailable

## Technical Flow

### Website Crawling Flow
```
User Input (URL) 
  → Browser Session Launch
  → Page Navigation & Extraction
  → Multi-page Crawling (if configured)
  → Content Aggregation
  → Analysis Summary Generation
  → LLM Prompt Construction (with metadata)
  → LLM API Call
  → JSON Parsing
  → SiteSummary Object Creation
  → PDF Report Generation
  → Response to User
```

### Text/Document Flow
```
User Input (Text/File)
  → Text Extraction (if file)
  → Content Type Detection (via prompt)
  → LLM Prompt Construction (context-aware)
  → LLM API Call (no schema validation)
  → JSON Parsing & Validation
  → SiteSummary Object Creation (dynamic sections)
  → PDF Report Generation
  → Response to User
```

## Key Design Decisions

### 1. **Why Dynamic Sections?
- Different content types need different analysis approaches
- Conversations require different structure than articles
- Provides more relevant and useful summaries

### 2. **Why Remove Schema for Text Summarization?
- Gemini API limitation with `additionalProperties`
- Prompt instructions are sufficient for structure
- More flexibility for LLM to adapt to content

### 3. **Why Keep Schema for Website Summaries?
- Fixed structure works well for websites
- Schema validation ensures consistency
- Better error handling with structured output

### 4. **Why Multiple LLM Providers?
- Flexibility and redundancy
- Different providers may have different strengths
- User can choose based on availability/preference

## Configuration

### Environment Variables
- `LLM_PROVIDER`: "gemini" or "grok"
- `GEMINI_API_KEY`: Google Gemini API key
- `GROK_API_KEY`: xAI Grok API key
- `CRAWL_MAX_PAGES`: Maximum pages to crawl (default: 3)
- `CRAWL_MAX_TOKENS`: Token limit for content (default: 4000)
- `CRAWL_TIMEOUT`: Timeout in seconds (default: 45)
- `PLAYWRIGHT_HEADLESS`: Run browser in headless mode (default: true)

## API Endpoints

1. **POST /api/analyze**: Analyze website URL
2. **POST /api/analyze-text**: Analyze raw text input
3. **POST /api/analyze-document**: Analyze uploaded document
4. **GET /api/stream**: Stream progress for website analysis
5. **GET /api/reports/{file_name}**: Download generated PDF

## Data Models

### SiteSummary
```python
{
    overview: str
    content_type: str
    sections: dict[str, list[str]]  # Dynamic sections
    highlights: list[str] | None  # Legacy support
    recommendations: list[str] | None  # Legacy support
}
```

### ServiceResult
```python
{
    url: str
    crawl: CrawlResult | None
    analysis: AnalysisSummary | None
    summary: SiteSummary
    pdf_path: str
}
```

## Performance Considerations

1. **Async Processing**: All I/O operations are async
2. **Streaming**: Website analysis uses SSE for real-time updates
3. **Caching**: Service instances cached with LRU cache
4. **Timeout Management**: Configurable timeouts for crawls and API calls
5. **Token Limits**: Content truncated to stay within token budgets

## Security Considerations

1. **File Upload Validation**: Path traversal protection
2. **API Key Security**: Stored in environment variables
3. **Content Sanitization**: Text cleaned before processing
4. **Error Message Sanitization**: No sensitive data in error messages

## Future Enhancements

1. Support for more document formats (DOCX, etc.)
2. Multi-language content detection
3. Caching of summaries for repeated content
4. Batch processing capabilities
5. Custom section templates per content type
6. Export to multiple formats (Markdown, HTML, etc.)

