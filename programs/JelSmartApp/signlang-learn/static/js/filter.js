document.addEventListener("DOMContentLoaded", () => {
    const storedData = JSON.parse(sessionStorage.getItem('lesson'));

    if (storedData && storedData.filter) {
        document.getElementById('searchInput').value = storedData.filter;
        document.getElementById("searchInput").dispatchEvent(new Event('input'));
    } else {
        const cards = document.querySelectorAll('#cardContainer .card');
        const lessonIds = [];

        cards.forEach(card => {
            const detailsButton = card.querySelector('#details');
            const id = parseInt(detailsButton.getAttribute("href").split('/').pop());
            lessonIds.push(id);
        });

        sessionStorage.setItem('lesson', JSON.stringify({
            ids: lessonIds,
            filter: null,
            page: document.documentURI
        }));
    }
});

document.getElementById('searchInput').addEventListener('input', function () {
    const filter = this.value.toLowerCase().trim();
    const cards = document.querySelectorAll('#cardContainer .card');
    const lessonIds = [];

    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        const detailsButton = card.querySelector('#details');

        if (text.includes(filter)) {
            card.style.display = '';

            const id = parseInt(detailsButton.getAttribute("href").split('/').pop());
            lessonIds.push(id);
        } else {
            card.style.display = 'none';
        }
    });

    sessionStorage.setItem('lesson', JSON.stringify({
        ids: lessonIds,
        filter: filter,
        page: document.documentURI
    }));
});
