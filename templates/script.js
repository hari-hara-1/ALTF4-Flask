document.getElementById('credit-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        if (key === "bnpl_used") {
            data[key] = value === "true";
        } else if (!isNaN(value) && value.trim() !== "") {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    });

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById('result').innerHTML = `<h2>Credit Score: ${result.credit_score}</h2>`;
    document.getElementById('suggestions').innerHTML = "<h3>Suggestions:</h3><ul>" + 
        result.suggestions.map(s => `<li>${s}</li>`).join('') + "</ul>";
});
