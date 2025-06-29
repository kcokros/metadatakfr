import streamlit as st
import json
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from openai import AzureOpenAI
import re
from typing import Dict, List, Optional
import traceback

# Page config
st.set_page_config(
    page_title="KFR Metadata Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://pbjai.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"

class KFRMetadataExtractor:
    def __init__(self, api_key: str):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=api_key,
            api_version=AZURE_OPENAI_API_VERSION
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using multiple methods for maximum accuracy"""
        text_content = ""
        
        try:
            # Method 1: PyMuPDF for clean text extraction
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.get_text()
            doc.close()
            
            # Method 2: pdfplumber for tables (if PyMuPDF text is insufficient)
            if len(text_content.strip()) < 1000:  # Fallback if text is too short
                pdf_file.seek(0)
                with pdfplumber.open(pdf_file) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text_content += f"\n--- Page {i + 1} (pdfplumber) ---\n"
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text
                        
                        # Extract tables
                        tables = page.extract_tables()
                        for j, table in enumerate(tables):
                            text_content += f"\n[Table {j+1} on Page {i+1}]\n"
                            for row in table:
                                if row:
                                    text_content += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
        
        return text_content
    
    def create_extraction_prompt(self, text: str, user_inputs: Dict) -> str:
        """Create optimized prompt for metadata extraction"""
        
        system_prompt = """
Anda adalah AI khusus untuk ekstraksi metadata dokumen Kajian Fiskal Regional (KFR) DJPb.

STRUKTUR STANDAR KFR:
- Cover: "KFR [Wilayah] Triwulan [I/II/III/IV] Tahun [YYYY]"
- Kata Pengantar: biasanya "Seulas Pinang" (Riau) atau judul khas daerah
- Ringkasan Eksekutif: berisi poin-poin kunci
- Dashboard: visualisasi data utama
- Bab I: Analisis Ekonomi Regional (makro ekonomi + kesejahteraan)
- Bab II: Analisis Fiskal Regional (APBN + APBD + box analysis)
- Bab III: Analisis Tematik (topik spesifik berbeda tiap wilayah)
- Bab IV: Kesimpulan dan Rekomendasi

PANDUAN EKSTRAKSI:

METADATA_UMUM:
- judul_dokumen: Dari cover page, format: "KFR DJPb [Wilayah] Triwulan [X] Tahun [YYYY]"
- periode: "Triwulan [I/II/III/IV] [YYYY]"
- wilayah: "Provinsi [Nama]" atau "DKI Jakarta"
- penyusun: Cari di halaman "The Team" atau kata pengantar, biasanya ["Tim RCE Kanwil [Wilayah]", "Seksi PPA"]
- reviewer: Dari kata pengantar, biasanya "Kepala Kanwil DJPb Provinsi [Wilayah]" 
- kategori: Selalu "KFR"
- tema: Dari judul Bab III Analisis Tematik atau subtitle di cover
- indikator_kunci: Dari Ringkasan Eksekutif, ambil 3-5 poin utama dengan angka
- tags: Kata kunci dari tema analisis dan isu utama yang dibahas
- tautan_file_pdf: null (akan diisi manual)
- tingkat_kerahasiaan: "Publik" (default untuk KFR)

METADATA_ANALISIS_KHUSUS:
- judul_dokumen: Sama dengan metadata_umum
- periode: Sama dengan metadata_umum
- tipe_analisis: "Fiskal & Ekonomi Regional" (standar)
- coverage_geografis: Dari teks, biasanya ["[X] Kabupaten/Kota di Provinsi [Nama]"]
- metodologi_khusus: Cari box analysis atau sub-bab metodologi di Bab II
- isu_khusus: Dari Bab III dan kesimpulan, fokus pada tema kebijakan
- topik_tematik: Dari judul dan sub-judul Bab III
- sumber_data: Cari disclaimer tabel "Sumber: [nama]", biasanya ["SPAN", "SAKTI", "APBD Kemendagri", "BPS", "BI", "ALCO"]

METADATA_TABEL_STRATEGIS:
- Prioritas tabel: APBN/APBD, pertumbuhan ekonomi, indikator fiskal
- Format ID: "Tabel_[Bab]_[nomor]" atau "Grafik_[Bab]_[nomor]"
- nama: Ambil dari judul tabel yang paling strategis
- deskripsi: Jelaskan isi tabel secara singkat
- wilayah: Sama dengan metadata umum
- periode: Sama dengan metadata umum
- kategori_tabel: Klasifikasi tabel (misal: "Realisasi APBN", "Indikator Ekonomi")
- tautan_sheet: null (akan diisi manual)
- kolom_tabel: Object dengan key-value kolom utama dan deskripsinya
- tag_analisis: Array tag relevan untuk tabel

CRITICAL INSTRUCTIONS:
- Pastikan konsistensi nama wilayah dan periode di semua metadata
- Cross-check angka indikator dengan sumber tabel
- Jika informasi tidak tersedia, gunakan null daripada menebak
- Output harus berupa JSON valid
- Fokus pada akurasi, bukan kelengkapan

OUTPUT FORMAT:
```json
{
  "metadata_umum": {...},
  "metadata_analisis_khusus": {...},
  "metadata_tabel_strategis": [{...}, {...}]
}
```
"""
        
        user_prompt = f"""
DOKUMEN KFR UNTUK DIANALISIS:

{text[:30000]}  # Limit untuk menghindari token limit

INFORMASI TAMBAHAN DARI USER:
- Wilayah: {user_inputs.get('wilayah', 'Tidak diisi')}
- Periode: {user_inputs.get('periode', 'Tidak diisi')}
- Catatan: {user_inputs.get('catatan', 'Tidak ada')}

TUGAS:
Ekstrak metadata sesuai format yang diminta. Prioritaskan akurasi dan konsistensi.
Jika ada informasi dari user yang bertentangan dengan dokumen, gunakan informasi dari dokumen.

OUTPUT JSON:
"""
        
        return system_prompt, user_prompt
    
    def extract_metadata(self, text: str, user_inputs: Dict) -> Dict:
        """Extract metadata using Azure OpenAI"""
        try:
            system_prompt, user_prompt = self.create_extraction_prompt(text, user_inputs)
            
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for accuracy
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_str = response_text
            
            # Parse JSON
            metadata = json.loads(json_str)
            return metadata
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {str(e)}")
            st.text("Raw response:")
            st.text(response_text)
            return None
        except Exception as e:
            st.error(f"Error calling Azure OpenAI: {str(e)}")
            return None

def validate_metadata(metadata: Dict) -> List[str]:
    """Validate extracted metadata and return list of issues"""
    issues = []
    
    if not metadata:
        return ["Metadata is empty or invalid"]
    
    # Check required fields in metadata_umum
    required_umum = ["judul_dokumen", "periode", "wilayah", "kategori"]
    umum = metadata.get("metadata_umum", {})
    
    for field in required_umum:
        if not umum.get(field):
            issues.append(f"Missing required field in metadata_umum: {field}")
    
    # Check consistency between sections
    if metadata.get("metadata_analisis_khusus"):
        analisis = metadata["metadata_analisis_khusus"]
        if umum.get("periode") != analisis.get("periode"):
            issues.append("Periode tidak konsisten antara metadata_umum dan metadata_analisis_khusus")
        if umum.get("judul_dokumen") != analisis.get("judul_dokumen"):
            issues.append("Judul dokumen tidak konsisten antara metadata_umum dan metadata_analisis_khusus")
    
    return issues

def main():
    st.title("📊 KFR Metadata Extractor")
    st.markdown("**Ekstraksi metadata otomatis dari dokumen Kajian Fiskal Regional (KFR) DJPb**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Azure OpenAI API Key",
            type="password",
            help="Masukkan API key Azure OpenAI Anda"
        )
        
        # Model settings
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max Tokens", 1000, 8000, 4000, 500)
        
        st.divider()
        
        # Optional user inputs
        st.subheader("📝 Informasi Tambahan (Opsional)")
        wilayah_input = st.text_input("Wilayah", placeholder="Contoh: Provinsi Riau")
        periode_input = st.text_input("Periode", placeholder="Contoh: Triwulan I 2025")
        catatan_input = st.text_area("Catatan", placeholder="Informasi tambahan yang ingin disampaikan")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload KFR PDF")
        uploaded_file = st.file_uploader(
            "Pilih file PDF KFR",
            type="pdf",
            help="Upload dokumen KFR dalam format PDF"
        )
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
            st.info(f"File size: {file_size:.2f} MB")
            
            if file_size > 50:
                st.warning("File size besar (>50MB). Processing mungkin memakan waktu lama.")
    
    with col2:
        st.subheader("🔧 Processing Controls")
        
        if st.button("🚀 Extract Metadata", type="primary", disabled=not (uploaded_file and api_key)):
            if not api_key:
                st.error("Silakan masukkan Azure OpenAI API Key di sidebar")
            elif not uploaded_file:
                st.error("Silakan upload file PDF terlebih dahulu")
            else:
                # Process the file
                with st.spinner("Mengekstrak text dari PDF..."):
                    extractor = KFRMetadataExtractor(api_key)
                    text_content = extractor.extract_text_from_pdf(uploaded_file)
                
                if not text_content.strip():
                    st.error("Gagal mengekstrak text dari PDF. Pastikan file tidak corrupt atau password-protected.")
                else:
                    st.success(f"Text extracted: {len(text_content)} characters")
                    
                    # Show preview of extracted text
                    with st.expander("👀 Preview Extracted Text"):
                        st.text_area("First 2000 characters:", text_content[:2000], height=200)
                    
                    # Prepare user inputs
                    user_inputs = {
                        "wilayah": wilayah_input,
                        "periode": periode_input,
                        "catatan": catatan_input
                    }
                    
                    # Extract metadata
                    with st.spinner("Mengekstrak metadata menggunakan AI..."):
                        metadata = extractor.extract_metadata(text_content, user_inputs)
                    
                    if metadata:
                        st.success("✅ Metadata berhasil diekstrak!")
                        
                        # Validate metadata
                        validation_issues = validate_metadata(metadata)
                        if validation_issues:
                            st.warning("⚠️ Issues detected:")
                            for issue in validation_issues:
                                st.write(f"- {issue}")
                        
                        # Display results
                        st.subheader("📋 Generated Metadata")
                        
                        # Create tabs for different metadata types
                        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metadata Umum", "🔍 Analisis Khusus", "📈 Tabel Strategis", "💾 JSON Export"])
                        
                        with tab1:
                            st.json(metadata.get("metadata_umum", {}))
                        
                        with tab2:
                            st.json(metadata.get("metadata_analisis_khusus", {}))
                        
                        with tab3:
                            tabel_strategis = metadata.get("metadata_tabel_strategis", [])
                            if tabel_strategis:
                                for i, tabel in enumerate(tabel_strategis):
                                    st.write(f"**Tabel {i+1}:**")
                                    st.json(tabel)
                            else:
                                st.info("Tidak ada tabel strategis yang ditemukan")
                        
                        with tab4:
                            st.subheader("Complete JSON Output")
                            json_output = json.dumps(metadata, indent=2, ensure_ascii=False)
                            st.text_area("JSON Output:", json_output, height=400)
                            
                            # Download button
                            filename = f"metadata_{uploaded_file.name.replace('.pdf', '')}.json"
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_output,
                                file_name=filename,
                                mime="application/json"
                            )
                    else:
                        st.error("❌ Gagal mengekstrak metadata. Silakan coba lagi atau periksa format dokumen.")

    # Footer
    st.divider()
    st.markdown("""
    **📌 Catatan Penggunaan:**
    - Pastikan dokumen KFR memiliki format standar DJPb
    - API key hanya digunakan untuk session ini dan tidak disimpan
    - Hasil ekstraksi dapat diedit manual sebelum digunakan
    - Untuk dokumen besar (>50MB), processing membutuhkan waktu lebih lama
    """)

if __name__ == "__main__":
    main()