import torch
import numpy as np
import streamlit as st
from model import Generator

# Load Generator
@st.cache(allow_output_mutation=True)
def load_generator(path='generator.pth', device='cpu'):
    G = Generator().to(device)
    G.load_state_dict(torch.load(path, map_location=device))
    G.eval()
    return G

def main():
    st.title('MNIST Digit Generator')
    st.write('Select a digit 0–9 and generate five handwritten-style samples.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = load_generator(device=device)

    digit = st.selectbox('Select Digit', list(range(10)))
    if st.button('Generate 5 Samples'):
        zs = torch.randn(5, 100, device=device)
        labels = torch.full((5,), digit, dtype=torch.long, device=device)
        with torch.no_grad():
            imgs = G(zs, labels).cpu().numpy()
        cols = st.columns(5)
        for i, col in enumerate(cols):
            img = ((imgs[i][0] + 1) * 127.5).astype('uint8')
            col.image(img, width=56)

if __name__ == '__main__':
    main()