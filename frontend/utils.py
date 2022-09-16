import streamlit as st


def image_grid(images, columns=2, show_caption: bool = True):
    st.header("Result Images:")
    c1, c2 = st.columns(columns)
    try:
        for i in range(0, len(images), columns):
            c1.image(images[i], caption=images[i] if show_caption else None)
            c2.image(images[i + 1], caption=images[i + 1] if show_caption else None)
    except IndexError:
        pass
