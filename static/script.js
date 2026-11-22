
async function predictNews() {

    const newsText = document.getElementById("newsText").value.trim();
    const button = document.getElementById("predictBtn");
    const result = document.getElementById("result");
    const category = document.getElementById("category");
    const error = document.getElementById("error");

    result.classList.add("hidden");
    error.textContent = "";

    if (!newsText) {
        error.textContent = "Please enter some news text.";
        return;
    }

    button.disabled = true;
    button.textContent = "Predicting...";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: newsText
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        category.textContent = data.category;
        result.classList.remove("hidden");

    } catch (err) {

        error.textContent = err.message;

    } finally {

        button.disabled = false;
        button.textContent = "Predict Category";

    }
}
