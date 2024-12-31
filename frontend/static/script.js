document.getElementById("imageForm").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent the default form submission

    const imageInput = document.getElementById("imageInput").files[0];
    const captionDisplay = document.getElementById("captionDisplay");

    if (imageInput) {
        captionDisplay.innerHTML = "Processing your image...";
        // Simulate sending the image to the server and receiving a caption
        setTimeout(() => {
            captionDisplay.innerHTML = "Caption: This is an example caption for the image.";
        }, 2000);
    } else {
        captionDisplay.innerHTML = "Please upload an image!";
    }
});
