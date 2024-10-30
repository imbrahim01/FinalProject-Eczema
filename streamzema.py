import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from skimage import io, color, measure, img_as_ubyte, feature, filters, exposure, morphology
from skimage.measure import regionprops_table
from io import BytesIO

# Judul Aplikasi
st.title("Pengolahan Citra Medika")

# Unggah file gambar
uploaded_file = st.file_uploader("Unggah file gambar eczema (format .jpg atau .png)", type=["jpg", "jpeg", "png"])

# Mengecek apakah file telah diunggah
if uploaded_file is not None:
    # Membaca gambar dan mengonversinya ke grayscale
    im = io.imread(BytesIO(uploaded_file.read()))
    img = color.rgb2gray(im)  # Convert image to grayscale
    img = img_as_ubyte(img)    # Convert to uint8

    # Sidebar menu
    with st.sidebar:
        selected = option_menu("Pengolahan Citra Medika", ["Home", "Encyclopedia", "Feature Extraction", "Chatbot"], default_index=0)

    # Home Page
    if selected == "Home":
        st.title('Final Project ✨')
        st.subheader("Anggota kelompok")
        group_members = [
            "Farhan Majid - 5023211049",
            "Leony Purba - 50232110",
            "Benedicta Sabdaningtyas - 50232110",
            "Adelia Safira - 50232110"
        ]
        for member in group_members:
            st.markdown(f"<p style='font-family:Georgia; color: black; font-size: 20px;'>{member}</p>", unsafe_allow_html=True)

    # Eczema Subacute Page
    elif selected == "Encyclopedia":
        st.markdown("<h1 style='text-align: center; color: red;'>🫀ENCYCLOPEDIA</h1>", unsafe_allow_html=True)
        questions = [
            ("1. Apa itu Eczema Subacute?", "Eczema subacute adalah bentuk eksim yang ditandai dengan gejala peradangan kulit yang sedang berlangsung, muncul setelah fase akut eksim. Pada tahap ini, gejala yang umum terjadi meliputi rasa gatal yang mengganggu, kemerahan dan pembengkakan pada kulit, kekeringan dan pengelupasan, serta lesi yang dapat mengeluarkan cairan atau membentuk keropeng saat mengering."),
            ("2. Apa gejala umum yang terjadi pada eczema subacute?", "Eczema subacute dapat dipicu oleh berbagai faktor, seperti alergen (serbuk sari atau debu), iritan (deterjen atau sabun), perubahan cuaca yang ekstrem, dan stres emosional. Pengobatan untuk eczema subacute biasanya melibatkan penggunaan krim atau salep kortikosteroid untuk mengurangi peradangan dan gatal, serta pelembap untuk menjaga kelembapan kulit. Penting juga untuk mengidentifikasi dan menghindari alergen atau iritan yang dapat memperburuk kondisi. Jika tidak ditangani dengan baik, eczema subacute dapat berkembang menjadi eczema kronis, sehingga mencari perawatan yang tepat dan mengikuti saran dokter sangatlah penting untuk mengelola gejala dan mencegah kekambuhan."),
        ]
        for title, description in questions:
            st.markdown(f"<p style='font-family:Georgia; color:yellow; font-size: 23px; text-align: left;'>{title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-family:Georgia; color:white; font-size: 20px; text-align: justify;'>{description}</p>", unsafe_allow_html=True)
        st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/fmurdUlmaIg" 
                    frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen></iframe>""", unsafe_allow_html=True)

    # Eczema Feature Extraction Page
    elif selected == "Feature Extraction":
        st.title("Eczema Image Feature Extraction 🧬")
        selected2 = option_menu(
            None, 
            ["Image", "Image Processing", "Edge Detection", "Image Segmentation", "Data"], 
            icons=['image', 'adjust', 'filter', 'table'], 
            menu_icon="cast", 
            default_index=0, 
            orientation="horizontal"
        )

        # Image Section
        if selected2 == "Image":
            st.subheader("Image Section")
            st.image(im, caption="Loaded Eczema Image", use_column_width=True)
            st.subheader("Grayscale Image (Converted to uint8)")
            st.image(img, caption="Grayscale Image", use_column_width=True)

        # Image Processing Section
        elif selected2 == "Image Processing":
            st.subheader("Pre-Processing")
            
            # Calculate Otsu's Threshold
            threshold = filters.threshold_otsu(img)
            st.write(f"Otsu's threshold value: {threshold}")

            # Apply Adaptive Histogram Equalization (AHE)
            img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255  
            img_hieq = img_hieq.astype('uint8')

            # Display original image with Otsu's contour and binary thresholded image
            fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
            ax[0].imshow(img, cmap='gray')
            ax[0].contour(img, levels=[threshold], colors='red')
            ax[0].set_title('Original Image with Otsu Contour')
            ax[1].imshow(img < threshold, cmap='gray')
            ax[1].set_title('Binary Image (Otsu Threshold Applied)')
            st.pyplot(fig)

            # Display AHE result
            st.subheader("Adaptive Histogram Equalization")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_hieq, cmap='gray')
            ax.set_title('Adaptive Histogram Equalization')
            st.pyplot(fig)

            # Otsu Thresholding on AHE result
            st.subheader("Otsu Thresholding on AHE Result")
            binary_image = img_hieq < filters.threshold_otsu(img_hieq)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(binary_image, cmap='gray')
            ax.set_title('Binary Image (Otsu Threshold on AHE)')
            st.pyplot(fig)

            # Remove small objects
            st.subheader("Remove Small Objects")
            only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(only_large_blobs, cmap='gray')
            ax.set_title('Binary Image with Small Objects Removed')
            st.pyplot(fig)

            # Fill small holes
            st.subheader("Fill Small Holes")
            only_large = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=100))
            image_segmented = only_large  # Save result for later use

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image_segmented, cmap='gray')
            ax.set_title('Binary Image with Small Holes Filled')
            st.pyplot(fig)

        # Edge Detection Section
        elif selected2 == "Edge Detection":
            st.subheader("Edge Detection Filters")
            
            # Re-apply segmentation if needed
            img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255
            binary_image = img_hieq < filters.threshold_otsu(img_hieq)
            only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
            only_large = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=100))
            image_segmented = only_large

            # Apply edge detection filters
            roberts = filters.roberts(image_segmented)
            sobel = filters.sobel(image_segmented)
            prewitt = filters.prewitt(image_segmented)
            canny = feature.canny(image_segmented, sigma=1)

            # Display edge detection results in a 2x2 grid
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            ax[0, 0].imshow(roberts, cmap='gray')
            ax[0, 0].set_title('Roberts')
            ax[0, 1].imshow(sobel, cmap='gray')
            ax[0, 1].set_title('Sobel')
            ax[1, 0].imshow(prewitt, cmap='gray')
            ax[1, 0].set_title('Prewitt')
            ax[1, 1].imshow(canny, cmap='gray')
            ax[1, 1].set_title(r'Canny $\sigma=1$')

            for a in ax.flat:
                a.axis('off')

            st.pyplot(fig)

        # Image Segmentation Section
        elif selected2 == "Image Segmentation":
            st.subheader("Contour Image")
            img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255
            binary_image = img_hieq < filters.threshold_otsu(img_hieq)
            only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
            only_large = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=100))
            image_segmented = only_large

            # Menampilkan gambar dengan kontur
            if 'image_segmented' in locals() and 'img' in locals():
                image_segmented = img_as_ubyte(image_segmented)
                threshold = filters.threshold_otsu(img) 
                
                fig, ax = plt.subplots()
                ax.imshow(image_segmented, cmap='gray')
                ax.contour(image_segmented, [threshold])
                st.pyplot(fig)

            # Tambahan kode untuk pewarnaan acak pada label
            lab_image = image_segmented
            rand_cmap = ListedColormap(np.random.rand(256, 3))  # Membuat colormap acak
            labels, nlabels = ndi.label(lab_image)

            labels_for_display = np.where(labels > 0, labels, np.nan)
            
            # Menampilkan gambar dengan label acak
            fig2, ax2 = plt.subplots()
            ax2.imshow(lab_image, cmap='gray')
            ax2.imshow(labels_for_display, cmap=rand_cmap)
            ax2.axis('off')
            ax2.set_title(f'Ezcema Subacute Labeled ({nlabels} labels)')
            st.pyplot(fig2)
            
            # Melakukan labeling pada objek yang ditemukan
            boxes = ndi.find_objects(labels)
            for label_ind, label_coords in enumerate(boxes):
                if label_coords is None:
                    continue  # Jika label tidak valid, lewati

                cell = lab_image[label_coords]
                
                # Filter objek berdasarkan ukuran
                cell_size = np.prod(cell.shape)

                if cell_size < 5000: 
                    lab_image = np.where(labels == label_ind + 1, 0, lab_image)
            
            # Regenerasi label setelah filter
            labels, nlabels = ndi.label(lab_image)
            st.write(f'Terdapat {nlabels} komponen / objek yang terdeteksi setelah filtering.')

            # Menampilkan subset dari objek yang terdeteksi
            fig3, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 6))
            for ii, obj_indices in enumerate(ndi.find_objects(labels)[5:11]):
                if obj_indices is not None:
                    cell = image_segmented[obj_indices]
                    axes[ii].imshow(cell, cmap='gray')
                    axes[ii].axis('off')
                    axes[ii].set_title(f'Label #{ii+1}\nUkuran: {cell.shape}')
            
            plt.tight_layout()
            st.pyplot(fig3)

            # Menjalankan labeling dan menganalisis properti region
            label_img = label(lab_image)
            regions = regionprops(label_img)
            
            # Menampilkan centroid dan orientasi pada gambar
            fig3, ax3 = plt.subplots()
            ax3.imshow(lab_image, cmap=plt.cm.gray)
            
            for props in regions:
                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

                # Plot centroid, orientasi, dan bounding box
                ax3.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax3.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax3.plot(x0, y0, '.g', markersize=15)

                # Plot bounding box
                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax3.plot(bx, by, '-b', linewidth=2.5)

            ax3.set_title("Centroid and Orientation of Labeled Regions")
            st.pyplot(fig3)

    # Data Extraction Section
    elif selected2 == "Data":
        st.subheader("Extracted Data")
        
        # Hitung properti menggunakan regionprops
        label_img = measure.label(img < filters.threshold_otsu(img))
        props = regionprops_table(label_img, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length'))
        df_new = pd.DataFrame(props)
        st.write("Newly Extracted Features:")
        st.write(df_new)

        # Path untuk menyimpan file Excel baru
        excel_path = r"extract_features.xlsx"  # Simpan di direktori kerja saat ini

        # Simpan data baru ke Excel (mode 'w' untuk membuat file baru)
        with pd.ExcelWriter(excel_path, mode='w', engine="openpyxl") as writer:
            df_new.to_excel(writer, index=False, sheet_name='New Features')

        # Tombol unduh untuk file Excel yang baru dibuat
        with open(excel_path, 'rb') as file:
            st.download_button(
                label="Download extracted features as Excel",
                data=file.read(),
                file_name="extract_features.xlsx",  # Nama file untuk unduhan
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Eczema Chatbot Page
elif selected == "Chatbot":
    st.title("Eczema Chatbot 🤖")

    # Initialize chat history if not already in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Predefined questions and answers for the chatbot
qa_pairs = {
    "What is eczema?": "Eczema is a condition that makes your skin red and itchy. It's common in children but can occur at any age.",
    "What are the symptoms of eczema?": "Symptoms include red to brownish-gray patches, itching (especially at night), and small, raised bumps that may leak fluid.",
    "How can I prevent eczema flare-ups?": "To prevent flare-ups, keep your skin moisturized, avoid triggers (like certain soaps, allergens, and stress), and avoid scratching.",
    "What treatments are available for eczema?": "Treatments may include moisturizers, steroid creams, and other topical or oral medications prescribed by a doctor.",
    "Can diet affect eczema?": "Certain foods may trigger eczema for some people, such as dairy, eggs, or nuts. It's best to consult a healthcare provider if you suspect food triggers.",
    "Is eczema contagious?": "No, eczema is not contagious. It’s a chronic condition often related to genetics and environmental triggers.",
    "Are there natural remedies for eczema?": "Yes, some natural remedies include coconut oil, aloe vera, and bathing in lukewarm water with added baking soda or oatmeal.",
    "Can stress worsen eczema?": "Yes, stress can exacerbate eczema symptoms. Stress management techniques, such as meditation or exercise, may help.",
    "What types of eczema are there?": "There are several types of eczema, including atopic dermatitis, contact dermatitis, dyshidrotic eczema, and seborrheic dermatitis.",
    "How can I manage itching caused by eczema?": "Try applying a cool, wet compress, using anti-itch creams, and keeping your skin moisturized. Avoid scratching to prevent skin damage.",
    "What causes eczema?": "The exact cause of eczema isn't known, but it's thought to be linked to an overactive immune system, genetics, environmental triggers, and skin barrier issues.",
    "Can eczema go away on its own?": "Eczema is a chronic condition, so it may not go away completely. However, symptoms can improve with age, especially in children.",
    "Is there a cure for eczema?": "Currently, there is no cure for eczema, but symptoms can be managed with treatments, lifestyle changes, and by avoiding triggers.",
    "How does eczema affect daily life?": "Eczema can impact daily activities due to discomfort from itching, pain, or visible skin changes, which may also affect confidence and mental well-being.",
    "Is eczema hereditary?": "Yes, eczema often runs in families. If parents have eczema, asthma, or allergies, their children may have a higher risk of developing eczema.",
    "Can eczema lead to other health issues?": "Eczema can increase the risk of skin infections due to scratching. It’s also associated with other conditions like asthma, allergies, and sleep disturbances.",
    "How do I know if I have eczema or psoriasis?": "Eczema and psoriasis may look similar but have different symptoms and triggers. Eczema often involves itchy, red patches, while psoriasis can be more scaly and less itchy. A dermatologist can provide a diagnosis.",
    "Can eczema affect mental health?": "Yes, living with eczema can lead to stress, anxiety, and depression due to discomfort and the condition’s visible impact on the skin.",
    "What fabrics should I avoid if I have eczema?": "Avoid rough fabrics like wool or synthetic materials that may irritate the skin. Opt for soft, breathable fabrics like cotton instead.",
    "Does eczema get worse in certain weather?": "Yes, eczema can worsen in cold, dry weather or in hot, humid conditions. Proper skincare routines can help manage flare-ups during extreme weather changes.",
    "Can eczema spread to other parts of the body?": "Eczema itself doesn’t spread like an infection, but symptoms can appear in multiple areas. Scratching or touching irritated skin can sometimes lead to new flare-ups elsewhere.",
    "How does eczema affect children differently?": "Children with eczema may experience more intense itching, which can disrupt sleep and affect school performance. They also have a higher risk of developing allergies and asthma.",
    "Can pets trigger eczema flare-ups?": "Yes, pet dander is a common allergen that can trigger eczema symptoms for some people. If you have pets, try to keep them out of the bedroom and clean frequently.",
    "How often should I moisturize if I have eczema?": "People with eczema should moisturize at least twice daily, and immediately after bathing, to help lock in moisture and strengthen the skin barrier.",
    "What should I do if my eczema becomes infected?": "If your eczema becomes red, warm, or begins to ooze, it could be infected. Consult a healthcare provider for appropriate treatment, which may include antibiotics.",
    "Are there any lifestyle changes that help with eczema?": "Yes, lifestyle changes like reducing stress, avoiding harsh soaps, wearing soft fabrics, and staying hydrated can help manage eczema.",
    "Can I wear makeup if I have eczema?": "Yes, but choose hypoallergenic, fragrance-free products, and always test a small area first. Avoid applying makeup during flare-ups, and keep your skin well-moisturized.",
    "What is the difference between eczema and dermatitis?": "Eczema is a type of dermatitis, which is an umbrella term for skin inflammation. Dermatitis includes eczema, contact dermatitis, and seborrheic dermatitis.",
    "Can I exercise with eczema?": "Yes, but try to avoid overheating and wear breathable clothing. Showering and moisturizing afterward can help prevent sweat-induced irritation.",
    "How can I tell if my child has eczema?": "In children, eczema often appears as red, itchy patches on the face, elbows, or knees. If you suspect eczema, consult a pediatrician or dermatologist for diagnosis."
}

# Function to get similar questions
def get_suggestions(query, choices, limit=5):
    # Get the closest matches
    suggestions = process.extract(query, choices, limit=limit)
    return [suggestion[0] for suggestion in suggestions]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display predefined questions for user to choose
st.write("### Ask me about Eczema:")
user_question = st.selectbox("Choose a question:", [""] + list(qa_pairs.keys()))

# If user selects a question, display answer
if user_question:
    # Display user question in chat
    with st.chat_message("user"):
        st.markdown(user_question)
    # Add user question to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Display bot response
    bot_response = qa_pairs.get(user_question, "I'm sorry, I don't have an answer for that question.")
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

