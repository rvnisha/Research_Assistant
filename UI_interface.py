import os
import re
import sys
import gradio as gr
from dotenv import load_dotenv
from orchestrator import ResearchAssistant
from tools.doc_retriever import DocRetriever
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
import tempfile

#CSS for UI
CUSTOM_CSS = """
/* 1) Main title (usually an <h1>) */
.gradio-container h1 {
    font-size: 36px !important;
    margin-bottom: 0.5em;
}

/* 2) Section headings (often <h3>) */
.gradio-container h3 {
    font-size: 24px !important;
    margin-top: 1em;
    margin-bottom: 0.25em;
}

/* 3) Body text & description (<p> tags) */
.gradio-container p {
    font-size: 20px !important;
    line-height: 1.5;
}

/* 4) Labels (for input fields) */
.gradio-container label {
    font-size: 20px !important;
}

/* 5) Buttons and inputs (if you want) */
.gradio-container button,
.gradio-container input,
.gradio-container textarea {
    font-size: 20px !important;
}
"""

##Report Generation of the Answer using ReportLab
def _apply_inline_styles(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",   r"<i>\1</i>", text)
    return text

def markdown_to_pdf(answer_md: str, sources: list, pdf_path: str):
    #Set up document and styles
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40, rightMargin=40,
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    # Adding styles Heading1 & Heading2 & Heading3 & Heading4
    styles["Heading1"].fontSize = 18
    styles["Heading1"].spaceAfter = 12

    styles["Heading2"].fontSize = 14
    styles["Heading2"].spaceAfter = 8

    styles["Heading3"].fontSize   = 12
    styles["Heading3"].spaceAfter = 6
    styles["Heading3"].leftIndent = 10

    styles["Heading4"].fontSize   = 12
    styles["Heading4"].spaceAfter = 6
    styles["Heading4"].leftIndent = 10

    if "MyBullet" not in styles:
        styles.add(ParagraphStyle(
            name="MyBullet",
            parent=styles["Normal"],
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=4,
        ))

    styles["Normal"].fontSize = 12
    styles["Normal"].leading = 14

    story = []

    # Converting markdown format to text
    list_buffer = []
    for line in answer_md.split("\n"):
        line = line.rstrip()
        line = _apply_inline_styles(line)
        if not line:
            if list_buffer:
                items = [
                    ListItem(Paragraph(item, styles["MyBullet"]))
                    for item in list_buffer
                ]
                story.append(ListFlowable(items, bulletType="bullet"))
                list_buffer = []
            story.append(Spacer(1, 8))
            continue

        if line.startswith("# "):
            # Heading1
            if list_buffer:
                items = [
                    ListItem(Paragraph(item, styles["MyBullet"]))
                    for item in list_buffer
                ]
                story.append(ListFlowable(items, bulletType="bullet"))
                list_buffer = []
            story.append(Paragraph(line[2:].strip(), styles["Heading1"]))

        elif line.startswith("## "):
            # Heading2
            if list_buffer:
                items = [
                    ListItem(Paragraph(item, styles["MyBullet"]))
                    for item in list_buffer
                ]
                story.append(ListFlowable(items, bulletType="bullet"))
                list_buffer = []
            story.append(Paragraph(line[3:].strip(), styles["Heading2"]))
        
        elif line.startswith("### "):
            # Heading3
            if list_buffer:
                items = [ListItem(Paragraph(i, styles["MyBullet"])) for i in list_buffer]
                story.append(ListFlowable(items, bulletType="bullet"))
                list_buffer = []
            story.append(Paragraph(line[4:].strip(), styles["Heading3"]))

        elif line.startswith("#### "):
            # Heading4
            if list_buffer:
                items = [ListItem(Paragraph(i, styles["MyBullet"])) for i in list_buffer]
                story.append(ListFlowable(items, bulletType="bullet"))
                list_buffer = []
            story.append(Paragraph(line[5:].strip(), styles["Heading4"]))


        elif line.startswith("- "):
            list_buffer.append(line[2:].strip())

        else:
            # normal paragraph
            if list_buffer:
                items = [
                    ListItem(Paragraph(item, styles["MyBullet"]))
                    for item in list_buffer
                ]
                story.append(ListFlowable(items, bulletType="bullet"))
                list_buffer = []
            story.append(Paragraph(line, styles["Normal"]))

    if list_buffer:
        items = [
            ListItem(Paragraph(item, styles["MyBullet"]))
            for item in list_buffer
        ]
        story.append(ListFlowable(items, bulletType="bullet"))

    # Adding Sources
    story.append(Spacer(1, 12))
    story.append(Paragraph("Sources:", styles["Heading2"]))
    for src in sources:
        story.append(Paragraph(src, styles["Normal"]))

    doc.build(story)

#Loading and verifying environment variables
print("🔄 Loading environment variables…")
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
serp_key   = os.getenv("SERPAPI_API_KEY")
if not openai_key or not serp_key:
    print("❌ ERROR: Missing API key(s). Check your .env file.", file=sys.stderr)
    sys.exit(1)
print("✅ API keys loaded.")

#Instantiating the assistant
print("🔧 Initializing ResearchAssistant…")
assistant = ResearchAssistant(max_results=5)
print("✅ Assistant ready.")


def answer_function(question: str, files):
    # If files uploaded → RAG over PDFs
    if files and len(files) > 0:
        print(f"💬 Received question: {question}")
        paths = [f.name for f in files]
        doc_ret = DocRetriever(chunk_size=1000, chunk_overlap=100, max_results=3)
        doc_ret.ingest(paths)

        answer, sources = doc_ret.retrieve(question)

    else:
        # If not uploaded Web based search
        print(f"💬 Received question: {question}")
        result = assistant.answer(question)
        answer = result['answer']
        sources  = result["sources"]
    
    sources_md = "\n".join(f"- {s}" for s in sources if s)
    print("💬Answer received")

    import tempfile
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    markdown_to_pdf(answer, sources, pdf_path)

    return answer, sources_md,pdf_path

##Gradio UI Interface

demo = gr.Interface(
    fn=answer_function,
    inputs=[
        gr.Textbox(lines=2, placeholder="Your question…", label="Question"),
        gr.Files(
            file_types=[".pdf"],
            label="Upload PDF(s)(optional)",
            file_count="multiple"
        )
    ],
    outputs=[gr.Markdown(label="Answer"),gr.Markdown(label="Sources"),gr.File(label="Download Answer as PDF")],
    title="🧠 Research Assistant",
    description=(
        "Upload your own PDF files to get a summary of the question you ask; "
        "otherwise, the assistant will search the web."
    ),
    flagging_mode ="never",
    css = CUSTOM_CSS
).queue()

if __name__ == "__main__":
    print("🚀 Launching Gradio app on http://localhost:7860 …")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
