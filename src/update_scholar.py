import os
from scholarly import scholarly
from bs4 import BeautifulSoup

AUTHOR_ID = "bXATl38AAAAJ"


PAPER_MAPPINGS = {
    "Biological Sex Determination in Cadavers": "cit-sex-det",
    "Automated identification of Ichneumonoidea wasps via YOLO": "cit-wasps",
    "Descriptor: Parasitoid Wasps and Associated Hymenoptera": "cit-dapwh",
    "A Leaf-Level Dataset for Soybean-Cotton": "cit-soybean",
    "Deep learning-based computer vision techniques for automated identification": "cit-dissertation",
    "The impact of feature scaling in machine learning": "cit-scaling",
    "A Synthetic Dataset for Manometry Recognition": "cit-manometry",
    "Breast Cancer Classification Using Gradient Boosting": "cit-breast-cancer",
    "Um estudo sobre algoritmos de Boosting": "cit-tcc"
}

def main():
    print("Buscando dados do Google Scholar...")
    author = scholarly.search_author_id(AUTHOR_ID)
    author = scholarly.fill(author, sections=['counts', 'publications'])
    
    total_citations = author.get('citedby', 0)
    total_publications = len(author.get('publications', []))
    print(f"Citações totais: {total_citations} | Publicações: {total_publications}")

    # Abre o HTML atual
    with open('index.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # 1. Atualiza totais (atualizando o texto e o data-target para a animação JS funcionar)
    span_citations = soup.find(id="total-citations")
    if span_citations:
        span_citations.string = str(total_citations)
        span_citations['data-target'] = str(total_citations)

    span_pubs = soup.find(id="total-publications")
    if span_pubs:
        span_pubs.string = str(total_publications)
        span_pubs['data-target'] = str(total_publications)

    # 2. Atualiza citações individuais por paper
    for pub in author.get('publications', []):
        title = pub['bib']['title']
        num_citations = pub.get('num_citations', 0)
        
        # Procura se o título do paper bate com nosso mapeamento
        for key_title, html_id in PAPER_MAPPINGS.items():
            if key_title.lower() in title.lower():
                span_pub = soup.find(id=html_id)
                if span_pub:
                    span_pub.string = str(num_citations)
                    print(f"Atualizado {html_id} para {num_citations} citações.")
                break

    # Salva o HTML modificado
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))
    print("index.html atualizado com sucesso.")

if __name__ == "__main__":
    main()
