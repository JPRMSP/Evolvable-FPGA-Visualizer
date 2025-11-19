import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time

st.set_page_config(layout="wide", page_title="Evolvable FPGA Visualizer")

# -------------------------------
# Utility functions
# -------------------------------

def random_genome(lut_count, lut_bits):
    # genome: flat binary array of length lut_count * lut_bits
    return np.random.randint(0,2, lut_count * lut_bits)

def genome_to_luts(genome, lut_bits):
    # returns array shape (lut_count, lut_bits)
    return genome.reshape(-1, lut_bits)

def lut_density(lut_bits_arr):
    # fraction of ones per LUT
    return lut_bits_arr.mean(axis=1)

def mutate_genome(genome, mut_rate=0.01):
    g = genome.copy()
    flips = np.random.rand(len(g)) < mut_rate
    g[flips] = 1 - g[flips]
    return g, flips.sum()

def crossover(a, b):
    p = random.randint(1, len(a)-1)
    child = np.concatenate([a[:p], b[p:]])
    return child

# routing helper: create random nets connecting LUT indices
def random_nets(lut_count, net_count):
    nets = []
    for _ in range(net_count):
        a, b = random.sample(range(lut_count), 2)
        nets.append((a,b))
    return nets

# coordinates of LUTs on 2D grid
def grid_coords(rows, cols):
    coords = []
    for r in range(rows):
        for c in range(cols):
            coords.append((c + 0.5, rows - r - 0.5))
    return coords

# plot grid heatmap of densities + routing lines + highlight
def plot_grid(density, rows, cols, selected_idx=None, nets=None, show_bit=False, lut_bits_arr=None):
    fig, ax = plt.subplots(figsize=(6,6))
    grid = density.reshape(rows, cols)
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=1, origin='upper')
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("LUT density (fraction of '1' bits)")

    # draw grid lines
    for x in range(-1, cols):
        ax.axvline(x + 0.5, color='white', lw=0.5, alpha=0.6)
    for y in range(-1, rows):
        ax.axhline(y + 0.5, color='white', lw=0.5, alpha=0.6)

    # overlay selected LUT box
    if selected_idx is not None:
        r = selected_idx // cols
        c = selected_idx % cols
        rect = plt.Rectangle((c, r), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # plot nets as lines between center coordinates
    if nets:
        coords = grid_coords(rows, cols)
        for a,b in nets:
            x1,y1 = coords[a]
            x2,y2 = coords[b]
            ax.plot([x1-0.5, x2-0.5],[y1-0.5, y2-0.5], lw=1.2, alpha=0.8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='density')
    plt.tight_layout()
    return fig

# plot bit pattern of a single LUT as small matrix
def plot_lut_bits(bitarray, lut_inputs):
    # bitarray length = 2**lut_inputs -> show as square if possible
    n = len(bitarray)
    # try to make shape near-square
    r = int(np.floor(np.sqrt(n)))
    c = int(np.ceil(n / r))
    arr = np.array(bitarray).reshape(r, c)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(arr, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"LUT bit pattern ({n} bits)")
    for (i,j), val in np.ndenumerate(arr):
        ax.text(j, i, int(val), ha='center', va='center', color='cyan', fontsize=9)
    plt.tight_layout()
    return fig

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧩 Evolvable FPGA Visualizer — LUT Genome & Routing Simulation")
st.markdown("""
This module visualizes an FPGA-style LUT fabric where **each LUT's configuration bits** form the genome.
You can mutate, animate, and observe routing between LUTs. Use it to demonstrate genome encoding,
dynamic reconfiguration and how LUT bitstreams change over time (intrinsic evolution visualization).
""")

# sidebar controls
with st.sidebar:
    st.header("Fabric / GA Controls")
    cols = st.number_input("Grid columns", min_value=2, max_value=16, value=8, step=1)
    rows = st.number_input("Grid rows", min_value=1, max_value=8, value=4, step=1)
    lut_inputs = st.selectbox("LUT inputs (k)", [2,3,4,5], index=2)  # 4-input LUT common
    lut_bits = 2 ** lut_inputs
    lut_count = rows * cols

    st.markdown(f"**LUTs:** {lut_count}   **bits per LUT:** {lut_bits}   **genome length:** {lut_count*lut_bits}")

    mut_rate = st.slider("Mutation rate (per bit)", 0.0, 0.5, 0.02, 0.01)
    animate_steps = st.slider("Animation steps", 1, 60, 12)
    anim_delay = st.slider("Animation delay (s)", 0.05, 1.0, 0.18, 0.01)

    net_count = st.slider("Number of nets (routing lines)", 0, lut_count//2, max(1, lut_count//4))

# stateful genome stored in session_state
if "genome" not in st.session_state:
    st.session_state.genome = random_genome(lut_count, lut_bits)
if "nets" not in st.session_state:
    st.session_state.nets = random_nets(lut_count, net_count)

# main layout: grid + controls + LUT details
col_left, col_right = st.columns([2,1])

with col_left:
    st.subheader("LUT Fabric View")
    density = lut_density(genome_to_luts(st.session_state.genome, lut_bits))
    fig = plot_grid(density, rows, cols, selected_idx=st.session_state.get("selected_idx", None),
                    nets=st.session_state.nets)
    fig_canvas = st.pyplot(fig)

    # control buttons
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Randomize Genome"):
        st.session_state.genome = random_genome(lut_count, lut_bits)
    if c2.button("Random Nets"):
        st.session_state.nets = random_nets(lut_count, net_count)
    if c3.button("Mutate Step"):
        st.session_state.genome, flips = mutate_genome(st.session_state.genome, mut_rate)
        st.success(f"Mutated {flips} bits")
    if c4.button("Animate Mutations"):
        # animate multiple mutation steps
        placeholder = st.empty()
        g = st.session_state.genome.copy()
        for i in range(animate_steps):
            g, flips = mutate_genome(g, mut_rate)
            density = lut_density(genome_to_luts(g, lut_bits))
            fig = plot_grid(density, rows, cols, selected_idx=st.session_state.get("selected_idx", None),
                            nets=st.session_state.nets)
            placeholder.pyplot(fig)
            time.sleep(anim_delay)
        st.session_state.genome = g
        placeholder.empty()
        st.success("Animation complete")

with col_right:
    st.subheader("Genome & LUT Inspector")
    # choose LUT index to inspect
    selected = st.number_input("Select LUT index", min_value=0, max_value=lut_count-1,
                               value=st.session_state.get("selected_idx", 0), step=1)
    st.session_state.selected_idx = selected

    lut_arr = genome_to_luts(st.session_state.genome, lut_bits)
    selected_bits = lut_arr[selected]
    st.markdown(f"**LUT #{selected} — fraction of ones:** {selected_bits.mean():.2f}")
    fig2 = plot_lut_bits(selected_bits, lut_inputs)
    st.pyplot(fig2)

    # manual bit editing: flip bit at chosen position
    st.markdown("**Manual bit editing (flip selected bit index)**")
    bit_index = st.number_input("Bit index in LUT", min_value=0, max_value=lut_bits-1, value=0, step=1)
    if st.button("Flip Selected Bit"):
        flat_idx = selected * lut_bits + bit_index
        st.session_state.genome[flat_idx] = 1 - st.session_state.genome[flat_idx]
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Routing Simulation Controls")
    if st.button("Highlight Nets (flash)"):
        placeholder_route = st.empty()
        for _ in range(4):
            figr = plot_grid(density, rows, cols, selected_idx=selected,
                             nets=st.session_state.nets)
            placeholder_route.pyplot(figr)
            time.sleep(0.18)
            placeholder_route.empty()
            time.sleep(0.08)
        st.experimental_rerun()

    st.markdown("**Nets (sample):**")
    sample_nets = st.session_state.nets[:min(6, len(st.session_state.nets))]
    st.write(sample_nets)

# footer: short explanation and tips
st.markdown("---")
st.markdown("""
**Notes & tips**
- Each LUT's configuration bits are the *genome*. You can mutate bits to simulate intrinsic evolution (reconfiguration).
- The heatmap encodes the density of 1s inside each LUT (a compact visualization of many bits).
- Routing lines are illustrative — to connect this with a GA-based mapping, you would:
  1. Define a target function / truth table.
  2. Define a fitness that measures how well the LUT fabric realizes the function (including routing cost).
  3. Evolve the genome with selection/crossover/mutation and map the best genome to the fabric for final hardware reconfiguration.
""")
