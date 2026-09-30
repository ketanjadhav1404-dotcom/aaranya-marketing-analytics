
import gradio as gr

from aaranya_agent import run_aaranya_agent


# ============================================================
# AARANYA CHAT FUNCTION
# ============================================================

def aaranya_chat(message, history):

    try:
        result = run_aaranya_agent(message)

        if result["route"] == "decision_engine":
            route_label = "💰 **Budget Optimization**"
        else:
            route_label = "🧠 **Knowledge Analysis**"

        return (
            f"{route_label}\n\n"
            f"{result['answer']}"
        )

    except Exception as e:

        return (
            "⚠️ **Aaranya could not process this question.**\n\n"
            f"Error: `{str(e)}`"
        )


# ============================================================
# RECOMMENDED QUESTIONS
# ============================================================

recommended_questions = [
    "What is adstock and why does Aaranya use it?",
    "What is the difference between attribution and causality?",
    "Why does Aaranya use P95 spending as the upper bound?",
    "What does the response curve tell us?",
    "I have ₹25,000 additional marketing budget. Where should I allocate it?",
    "I have ₹50,000 additional marketing budget. Where should I allocate it?",
]


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """
footer {
    display: none !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#recommended-title {
    margin-top: 10px;
    margin-bottom: 8px;
}

#recommended-subtitle {
    margin-bottom: 10px;
    color: #777;
}

.recommended-btn {
    min-height: 55px !important;
    white-space: normal !important;
    text-align: left !important;
}
"""


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(
    title="Aaranya Marketing Analytics"
) as demo:

    gr.Markdown(
        "# 🌿 Aaranya Marketing Analytics"
    )

    gr.Markdown(
        "**AI-powered marketing decision support**\n\n"
        "Ask Aaranya about marketing performance, attribution, "
        "MMM, adstock, response curves, or where to allocate "
        "the next ₹ of marketing budget."
    )

    gr.Markdown(
        "### 💡 Recommended questions",
        elem_id="recommended-title"
    )

    gr.Markdown(
        "Click a question to send it directly to Aaranya.",
        elem_id="recommended-subtitle"
    )

    # --------------------------------------------------------
    # Recommended question buttons
    # --------------------------------------------------------

    with gr.Row():

        q1 = gr.Button(
            "📊 What is adstock and why does Aaranya use it?",
            elem_classes="recommended-btn"
        )

        q2 = gr.Button(
            "🔎 What is the difference between attribution and causality?",
            elem_classes="recommended-btn"
        )

        q3 = gr.Button(
            "📐 Why does Aaranya use P95 spending as the upper bound?",
            elem_classes="recommended-btn"
        )

    with gr.Row():

        q4 = gr.Button(
            "📈 What does the response curve tell us?",
            elem_classes="recommended-btn"
        )

        q5 = gr.Button(
            "💰 I have ₹25,000 additional budget. Where should I allocate it?",
            elem_classes="recommended-btn"
        )

        q6 = gr.Button(
            "💰 I have ₹50,000 additional budget. Where should I allocate it?",
            elem_classes="recommended-btn"
        )


    # --------------------------------------------------------
    # Chat interface
    # --------------------------------------------------------

    textbox = gr.Textbox(
        placeholder="Ask Aaranya a question...",
        label="Your question",
        lines=2
    )

    chatbot = gr.Chatbot(
        height=500,
        label="Aaranya"
    )


    # --------------------------------------------------------
    # Send message
    # --------------------------------------------------------

    def respond(message, history):

        if not message or not message.strip():
            return history, ""

        result = aaranya_chat(
            message,
            history
        )

        history = history + [
            {
                "role": "user",
                "content": message
            },
            {
                "role": "assistant",
                "content": result
            }
        ]

        return history, ""


    textbox.submit(
        respond,
        inputs=[textbox, chatbot],
        outputs=[chatbot, textbox]
    )


    # --------------------------------------------------------
    # Recommended question behaviour
    # --------------------------------------------------------

    buttons = [
        q1, q2, q3, q4, q5, q6
    ]

    for button, question in zip(
        buttons,
        recommended_questions
    ):

        button.click(
            respond,
            inputs=[
                gr.State(question),
                chatbot
            ],
            outputs=[
                chatbot,
                textbox
            ]
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        theme=gr.themes.Soft(),
        css=custom_css,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7862
    )
