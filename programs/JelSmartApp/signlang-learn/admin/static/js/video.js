document.getElementById("image").addEventListener("change", function(event) {
      const preview = document.getElementById("image-preview");
      preview.innerHTML = "";

      const file = event.target.files[0];
      if (file) {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        preview.appendChild(img);
      }
    });

    document.getElementById("video").addEventListener("change", function(event) {
      const preview = document.getElementById("video-preview");
      preview.innerHTML = "";

      const file = event.target.files[0];
      if (file) {
        const video = document.createElement("video");
        video.src = URL.createObjectURL(file);
        video.controls = true;
        preview.appendChild(video);
      }
    });