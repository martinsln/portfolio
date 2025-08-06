function toggleMenu() {
  document.getElementById('sideMenu').classList.toggle('open');
}

document.querySelectorAll("[data-carousel]").forEach(carousel => {
  const images = carousel.querySelectorAll(".carousel-image");
  const dotsContainer = carousel.querySelector(".carousel-dots");
  let current = 0;

  // Créer les petits ronds (dots)
  images.forEach((_, index) => {
    const dot = document.createElement("span");
    dot.addEventListener("click", () => {
      current = index;
      show(current);
    });
    dotsContainer.appendChild(dot);
  });

  const dots = dotsContainer.querySelectorAll("span");

  const show = (index) => {
    images.forEach((img, i) => {
      img.classList.toggle("show", i === index);
      dots[i].classList.toggle("active", i === index);
    });
  };

  carousel.querySelector(".prev").addEventListener("click", () => {
    current = (current - 1 + images.length) % images.length;
    show(current);
  });

  carousel.querySelector(".next").addEventListener("click", () => {
    current = (current + 1) % images.length;
    show(current);
  });

  show(current);
});

function openLightbox(src) {
  const lightbox = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  img.src = src;
  lightbox.style.display = "flex";
}

function closeLightbox() {
  document.getElementById("lightbox").style.display = "none";
}  

  window.addEventListener('load', () => {
    // Cacher le loader
    document.getElementById('loader').style.display = 'none';
    // Afficher le contenu
    document.getElementById('content').style.display = 'block';
  });

