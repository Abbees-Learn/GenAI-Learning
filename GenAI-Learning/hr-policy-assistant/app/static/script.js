
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const messages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");
const status = document.getElementById("status");
const welcome = document.getElementById("welcome");

function addMessage(text, role) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    // textContent prevents document content from injecting HTML.
    message.textContent = text;
    messages.appendChild(message);

    messages.scrollTop = messages.scrollHeight;
    return message;
}

function showSources(parent, sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        return;
    }

    const details = document.createElement("details");
    details.className = "sources";

    const summary = document.createElement("summary");
    summary.textContent = `View sources (${sources.length})`;
    details.appendChild(summary);

    for (const source of sources) {
        const item = document.createElement("div");
        item.className = "source-item";

        const title = document.createElement("strong");
        title.textContent =
            `${source.document} — Page ${source.page}`;

        const excerpt = document.createElement("p");
        excerpt.textContent = source.excerpt;

        item.append(title, excerpt);
        details.appendChild(item);
    }

    parent.appendChild(details);
}

async function askQuestion(question) {
    if (!question.trim() || sendButton.disabled) {
        return;
    }

    welcome.hidden = true;
    addMessage(question, "user");

    input.value = "";
    input.style.height = "auto";
    sendButton.disabled = true;
    status.textContent = "Searching your documents and generating an answer...";

    const loading = addMessage("Thinking...", "assistant");

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question.trim()
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "The server could not process the question."
            );
        }

        loading.textContent = data.answer || "No answer returned.";
        showSources(loading, data.sources);

    } catch (error) {
        loading.textContent = `Error: ${error.message}`;
        console.error("API request failed:", error);
    } finally {
        sendButton.disabled = false;
        status.textContent = "";
        input.focus();
        messages.scrollTop = messages.scrollHeight;
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(input.value);
});

input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height =
        `${Math.min(input.scrollHeight, 140)}px`;
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        askQuestion(button.dataset.question || "");
    });
});
