// Function to load the fit options form
function loadFitOptions() {
    const fitOptionsContainer = document.getElementById("fitOptionsContainer");
    if (fitOptionsContainer.innerHTML.trim() === "") {
      fetch("templates/training_arguments.html")
        .then(response => response.text())
        .then(html => {
          fitOptionsContainer.innerHTML = html;
          fitOptionsContainer.style.display = "block";
        })
        .catch(err => console.error("Failed to load fit options form:", err));
    } else {
      fitOptionsContainer.style.display = fitOptionsContainer.style.display === "none" ? "block" : "none";
    }
  }
  