/**
 * Sentence display and keyboard controls for ASL Recognition app.
 */

class SentenceManager {
    constructor() {
        this.sentenceBox = document.getElementById('sentenceBox');
        this.btnClear = document.getElementById('btnClear');
        this.btnSpace = document.getElementById('btnSpace');
        this.btnDel = document.getElementById('btnDel');
        this.placeholder = 'Start signing to build a sentence...';

        this.bindEvents();
    }

    bindEvents() {
        this.btnClear.addEventListener('click', () => {
            this.clear();
        });

        this.btnSpace.addEventListener('click', () => {
            this.addSpace();
        });

        this.btnDel.addEventListener('click', () => {
            this.deleteLast();
        });

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === ' ' && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                this.addSpace();
            } else if (e.key === 'Backspace') {
                e.preventDefault();
                this.deleteLast();
            } else if (e.key === 'c' || e.key === 'C') {
                e.preventDefault();
                this.clear();
            }
        });
    }

    display(text) {
        if (!text || text.trim() === '') {
            this.sentenceBox.textContent = this.placeholder;
            this.sentenceBox.style.color = '#555';
        } else {
            this.sentenceBox.textContent = text;
            this.sentenceBox.style.color = '#ffffff';
        }
    }

    clear() {
        this.sentenceBox.textContent = this.placeholder;
        this.sentenceBox.style.color = '#555';
        if (this.onSentenceUpdate) {
            this.onSentenceUpdate('clear', '');
        }
    }

    addSpace() {
        if (this.onSentenceUpdate) {
            this.onSentenceUpdate('space', '');
        }
    }

    deleteLast() {
        if (this.onSentenceUpdate) {
            this.onSentenceUpdate('del', '');
        }
    }

    updateFromServer(sentence) {
        this.display(sentence);
    }
}
