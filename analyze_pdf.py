from docling.document_converter import DocumentConverter

# Analyze the PDF structure
converter = DocumentConverter()
result = converter.convert(r'c:\Users\HP\Downloads\Reading4-NLP.pdf')

# Get markdown content
md_content = result.document.export_to_markdown()

print("=== PDF ANALYSIS ===")
print(f"Total characters: {len(md_content)}")
print(f"Estimated words: {len(md_content.split())}")
print(f"Estimated tokens: {len(md_content) // 4}")

print("\n=== CONTENT STRUCTURE (First 3000 chars) ===")
print(md_content[:3000])

print("\n=== CONTENT STRUCTURE (Sample from middle) ===")
print(md_content[len(md_content)//2:len(md_content)//2 + 1500])

# Check for headers/sections
lines = md_content.split('\n')
headers = [line for line in lines if line.startswith('#')]
print(f"\n=== DETECTED HEADERS ({len(headers)} total) ===")
for header in headers[:20]:  # Show first 20 headers
    print(header)
