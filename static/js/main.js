// Select the register button element by its id
var registerButton = document.getElementById("register");

// Define a function that will register the user
function registerUser() {
    // Get the user input from a prompt window
    var userName = prompt("Enter your name");
    // Check if the user input is not empty
    if (userName) {
        // Display a message that the user is registered
        alert("You are registered as " + userName);
    }
}

// Add an event listener that will call the registerUser function when the register button is clicked
registerButton.addEventListener("click", registerUser);