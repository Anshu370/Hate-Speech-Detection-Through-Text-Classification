import google.generativeai as genai

# Configure API key
genai.configure(api_key='AIzaSyAHYEa0F8DA6vy9JZ9WLnVkbVUepR3VA1Q')

# Create GenerativeModel instance
model = genai.GenerativeModel('gemini-1.5-flash')

text = input("Enter your text\t")
response = model.generate_content(f"Is this hate speech or not\"{text}\"")
print(response.text)