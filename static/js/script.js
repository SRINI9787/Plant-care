const input = document.querySelector("#image-input");
const form = document.querySelector("#upload-form");

if (input && form) {
  const uploadZone = document.querySelector("#upload-zone");
  const preview = document.querySelector("#image-preview");
  const title = document.querySelector("#upload-title");
  const subtitle = document.querySelector("#upload-subtitle");
  const error = document.querySelector("#upload-error");
  const button = document.querySelector("#analyze-button");
  const maxBytes = 8 * 1024 * 1024;
  const validTypes = ["image/jpeg", "image/png"];
  let previewUrl = null;

  function showError(message) {
    error.textContent = message;
    error.hidden = false;
  }

  function clearError() {
    error.textContent = "";
    error.hidden = true;
  }

  function validateFile(file) {
    if (!file) return "Choose a JPG or PNG image first.";
    if (!validTypes.includes(file.type)) return "That file type isn't supported. Choose a JPG or PNG image.";
    if (file.size > maxBytes) return "That image is larger than 8 MB. Choose a smaller file.";
    return "";
  }

  function displayFile(file) {
    clearError();
    const problem = validateFile(file);
    if (problem) {
      input.value = "";
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = null;
      preview.hidden = true;
      uploadZone.classList.remove("has-preview");
      title.textContent = "Drop a leaf photo here";
      subtitle.innerHTML = 'or <span class="browse-link">browse files</span> on your device';
      showError(problem);
      return;
    }

    title.textContent = file.name;
    subtitle.textContent = "Looking good — ready for a plant check";
    uploadZone.classList.add("has-preview");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
  }

  input.addEventListener("change", () => displayFile(input.files[0]));

  ["dragenter", "dragover"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      uploadZone.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      uploadZone.classList.remove("is-dragging");
    });
  });

  uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    displayFile(file);
  });

  form.addEventListener("submit", (event) => {
    clearError();
    const problem = validateFile(input.files[0]);
    if (problem) {
      event.preventDefault();
      showError(problem);
      return;
    }
    button.disabled = true;
    button.innerHTML = '<span class="spinner" aria-hidden="true"></span> Checking your leaf…';
    form.setAttribute("aria-busy", "true");
  });
}
