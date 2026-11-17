# Đớp Tool — tối ưu hoá kết cấu thép bằng Phép tính Biến phân

Chạy: `pip install -r requirements.txt` rồi `streamlit run web_app.py`. Kiểm thử: `python -m pytest -q tests`.

## Cấu trúc
| File | Vai trò |
|---|---|
| `fem_core.py` | Core FEM của bạn, **giữ nguyên** (ma trận phần tử, ma trận xoay, tải tương đương, nội lực). |
| `opt_core.py` | Lõi tối ưu: biến phân Euler–Lagrange + FEM-OC/KKT. |
| `web_app.py` | Giao diện Streamlit: Lý thuyết · Cột · Dầm vút thon · Toàn khung/dàn · Động lực học SDOF. |
| `dynamic_core.py` | Dao động tự do SDOF, tích phân Newmark, phổ đáp ứng đàn hồi và nhà cắt MDOF. |
| `opensees_core.py` | Khung không gian 3D OpenSeesPy và ba thủ tục động đất đàn hồi. |
| `opensees_ui.py` | Tab OpenSees 3D, nhập phổ/bản ghi và so sánh đáp ứng ELF/MRSA/THA. |
| `canvas/index.html` | Canvas vẽ kết cấu (nút/thanh/gối). |
| `tests/test_core.py` | Kiểm chứng số học. |

## Trực quan hoá trước / sau tối ưu
Cột, dầm và từng thanh khung/dàn đều có hình **trước (nét đứt)** so với **sau (nét liền, tô màu)**: xanh lá = vật liệu bỏ được, cam = chỗ dày hơn mốc.
Mốc cột: cùng P_cr hoặc cùng thể tích (chọn được). Mốc dầm và khung/dàn: tiết diện đều nhỏ nhất thoả cùng ràng buộc.
Kích thước vẽ ∝ A^γ (γ của họ tiết diện; γ = 0 như chữ I cao cố định thì vẽ ∝ A). Với cùng thể tích, nghiệm tối ưu của cột luôn phình ra ngoài mốc ở giữa nhịp.

## Phương pháp
1. **Cấu kiện biến thiên A(x), I = α·Aⁿ.**
   - *Cột:* Euler–Lagrange cho `A ∝ |w|^(2/(n+1))`, nghiệm đóng qua hàm Beta không đầy đủ; cột khớp–khớp n=2 cho tỷ số tải tới hạn 4/3 (Keller).
   - *Dầm tĩnh định:* F không chứa A′ nên Euler–Lagrange suy biến thành phương trình đại số `A = [λ·n·M·m̄/(Eα)]^(1/(n+1))`; λ giải từ ràng buộc võng.
   - *Dầm siêu tĩnh:* M phụ thuộc A nên không có nghiệm đóng, dùng FEM + KKT rời rạc.
2. **Toàn khung/dàn:** FEM (tái dùng `fem_core`) + ràng buộc bền/ổn định Euler (FSD) + ràng buộc chuyển vị (OC, rồi SLSQP với đạo hàm liên hợp).
3. **Động lực học:** dao động tự do SDOF tuyến tính; đáp ứng SDOF đàn hồi và nhà cắt nhiều tầng song tuyến tính có dẻo động học bằng Newmark–Newton; phổ Sd/Sa; lệch tầng và độ dẻo theo thời gian; mốc vượt giới hạn biến dạng quy đổi theo vật liệu đã chọn. Mô hình khung không gian 3D đàn hồi rút gọn độ cứng tầng từ hình học/tiết diện; giao diện có hoạt ảnh dao động 3D. Chức năng tối ưu sơ bộ dò mức tăng tiết diện nhỏ nhất đạt mục tiêu lệch tầng trong bản ghi. Các bộ giải lấy cảm hứng từ ví dụ SDOF/OpenSees trong OPENSEES_SALAR-main và cài độc lập bằng NumPy.

4. **Phương pháp động đất trong mô hình khung nhiều tầng:** tĩnh lực ngang tương đương / Equivalent Lateral Force (ELF) dùng Sa(T₁) nhập từ phổ thiết kế; phổ phản ứng dạng dao động / Modal Response Spectrum Analysis (MRSA) đọc phổ T–Sa CSV và tổ hợp SRSS/CQC; lịch sử thời gian / Time-History Analysis (THA) nhận chuỗi gia tốc nền hoặc tín hiệu điều hòa minh họa. Cả ba được tính trên shear-building rút gọn từ FEM; ELF/MRSA đàn hồi, THA phi tuyến giảm bậc. QCVN 02 chỉ tra thông số nguy hiểm theo địa điểm, không tự tạo accelerogram hay phổ thiết kế hoàn chỉnh.

ELF/MRSA có sẵn **phổ đàn hồi mẫu Type 1**: lấy a₍gR₎ theo địa danh từ QCVN 02:2021/BXD, sau đó áp dụng hệ số nền Type 1, hệ số tầm quan trọng và tỷ số cản theo dạng TCVN 9386-1:2025 / EN 1998-1. Phổ mặc định q=1 để minh họa đàn hồi; khi thiết kế cần dùng phổ giảm theo q và tham số dự án, hoặc nạp/nhập phổ thiết kế riêng.

## Động đất OpenSees 3D
Tab **Động đất · OpenSees 3D** dựng khung dầm-cột không gian với diaphragm cứng, khối lượng sàn tập trung, chân cột ngàm và phần tử `elasticBeamColumn`. ELF dùng phân tích tĩnh; MRSA dùng mode shape/eigen của OpenSees và tổ hợp SRSS/CQC; THA dùng `UniformExcitation`, `Path` time series, cản Rayleigh và Newmark. Kết quả THA được trình bày bằng gia tốc nền, profile lệch tầng, quỹ đạo V–u và hoạt ảnh biến dạng theo thời gian. Ba phương pháp cùng dùng mô hình đàn hồi để so sánh đáp ứng; phần trăm trong bảng là chênh lệch so với THA, không phải sai số độ chính xác. Mô hình này chưa xét phi tuyến vật liệu, tải trọng trọng lực/P-Delta, tương tác đất-móng hoặc kiểm tra thiết kế đầy đủ.

Quy trình ban đầu tham khảo ví dụ của Salar Delavar Ghashghaei trong [OPENSEES_SALAR](https://github.com/salardelavar/OPENSEES_SALAR) (MIT License). Backend gọi trực tiếp OpenSeesPy 3.8; bộ sinh khung không gian, giao diện và bước so sánh là phần phát triển riêng của Đớp Tool. `pip install -r requirements.txt` cài backend. Trên Windows cần Python 3.12 64-bit vì wheel OpenSeesPy liên kết với `python312.dll`; `Dop_Tool.bat` tạo môi trường `.venv312` để dùng đúng runtime.

**Phạm vi động lực học hiện tại:** FEM khung đàn hồi 3D dùng độ cứng quy đổi từ cốt dọc, sau đó quy đổi sang mô hình nhà cắt phi tuyến một bậc tự do mỗi tầng; sức kháng chảy uốn ước tính từ bố trí cốt dọc; đai dùng trong chỉ số kiềm chế biến dạng giản lược. Chân cột khóa 6 DOF; lực cắt đáy lấy từ lực cắt tầng 1. Sức kháng đai cổ cột chỉ là thành phần thép ngang ước tính, không gồm bê tông, neo, mặt tiếp xúc, đài/móng hay nền đất. Đây chưa phải mô hình thớ RC hay thiết kế liên kết móng. TCVN 5574:2018 là tiêu chuẩn vật liệu/kết cấu bê tông; tiêu chuẩn kháng chấn mới TCVN 9386-1:2025 đã thay thế bản 2012. Thời điểm vượt ngưỡng chỉ có nghĩa trong bản ghi và giả thiết đã nhập, không phải dự báo sụp đổ được chứng nhận.

## Kiểm chứng đã làm
- Cột đều: FEM khớp công thức Euler (sai số < 1e-4) cho 3 sơ đồ.
- Cột tối ưu khớp–khớp: tỷ số 4/3 chính xác; FEM trên profile rời rạc lệch < 1% so với nghiệm đóng.
- Dầm tựa đơn: thể tích FEM-OC lệch < 2% so với nghiệm đóng E–L.
- Bộ giải nhanh trùng `fem_core.solve_plane_frame` (sai khác tương đối ≤ 1e-10).
- Phản lực cân bằng ΣFx, ΣFy, ΣM ≈ 1e-11.
- **Dàn 10 thanh chuẩn (Schmit):** 5076.7 lb so với cực trị đã biết 5060.85 lb (+0.3%).

## Ví dụ slide: dàn 10 thanh
Trong tab **Toàn khung / dàn**, chọn preset **Dàn 10 thanh chuẩn (slide)** rồi nạp mẫu. Hình học gồm hai khoang 360 in × 360 in, hai tải đứng 100 kip tại các nút đáy giữa/phải; giới hạn chuyển vị là 2 in tại các nút tự do. Tham số chuẩn được đổi sang kN–m: E = 10,000 ksi, [σ] = 25 ksi, A_min = 0.1 in² và trọng lượng riêng = 0.1 lb/in³. Trọng lượng tham chiếu khoảng 5,060.85 lb; test cho phép 4,900–5,200 lb để bao phủ sai khác của thuật toán OC/FSD.

## Giới hạn đã biết
- Dàn/khung **siêu tĩnh** có thể dừng ở điểm dừng cục bộ cách cực trị vài phần nghìn (cận bền A_lb được cố định theo nội lực mỗi vòng, không tính độ nhạy nội lực theo A).
- Euler–Bernoulli, đàn hồi; chưa xét P-Δ, ổn định cục bộ, mỏi, tải bản thân, liên kết. Tiết diện thực cần làm tròn theo thép hình rồi kiểm tra lại theo TCVN/AISC.

## Lưu ý về `fem_core.py`
`solve_plane_frame` báo **phản lực sai dấu** (lực xuống 10 kN ở đầu console cho Ry = −10, và phần "Kiểm tra ΣFy" ra −20 thay vì 0). Core được giữ nguyên; lõi tối ưu dùng phản lực riêng đã đúng dấu.

## Biểu đồ tải trọng · lực cắt · mô men · chuyển vị (ghép từ web cũ)
- `legacy_plots.py`: trích nguyên văn các hàm vẽ từ `web_app_v2.py` (sơ đồ tải, SFD, BMD, đường đàn hồi, biểu đồ khung).
- `viz_results.py`: cầu nối giữa kết quả tối ưu và các hàm vẽ cũ; nét "trước tối ưu" là nét đứt **trắng** trên nền tối (đen trên nền sáng).
- Tab Dầm: lưới 2×2 (tải / V / M / võng) + phản lực. Tab Khung/dàn: các tab Sơ đồ tải, Nội lực (M/V/N), Chuyển vị.
- Quy ước: tải hướng xuống > 0; V dương khi phần bên trái đẩy lên; M > 0 căng thớ dưới (vẽ xuống dưới); võng xuống > 0.
- Lưu ý: `fem_core.solve_plane_frame` trả dấu phản lực ngược (giữ nguyên core); `opt_core.reactions()` đã hiệu chỉnh. Cột chưa có biểu đồ V/M.

## Kiểm toán thuật toán (10/2026)
Các phép kiểm độc lập nằm trong `tests/audit/` (so với SLSQP viết riêng, đạo hàm sai phân, nhiều điểm khởi tạo):
- **Cột:** hệ số tăng tải khớp–khớp/console khớp nghiệm đóng (4/3 với n=2; 1.4072 với n=3; 1.2159 với n=1); nhiễu ngẫu nhiên bảo toàn thể tích không làm tăng P_cr.
- **Dầm:** 5 sơ đồ (tựa đơn, console, ngàm–gối, hai ngàm) khớp SLSQP độc lập tới 5 chữ số; nghiệm đóng E–L hội tụ bậc 1 theo mịn lưới (4.9% → 2.4% → 1.2% với 30/60/120 phần tử).
- **Khung/dàn:** Warren (có/không Euler), khung cổng, khung 2 tầng, dàn 10 thanh khớp SLSQP đầy đủ ràng buộc (sai ≤ 0.01%).
### Lỗi đã sửa
1. **Cột hai đầu ngàm:** profile đóng đơn-mode có A=0 tại L/4, 3L/4 ⇒ cơ cấu; FEM chỉ ×0.985 (kém cột đều). Nay dùng nghiệm bimodal (λ₁=λ₂) bằng SLSQP (×1.31 ở N=60); chế độ P_req được hiệu chỉnh V để FEM đạt đúng P_req.
2. **Báo nhầm "cơ cấu"** ở dầm lưới mịn (≥ ~200 phần tử; slider cho phép tới 200 + điểm tải): ngưỡng λmin/λmax 1e-9 → 1e-13 (cơ cấu thật ≈ 1e-17).
3. **Cực tiểu cục bộ:** dàn siêu tĩnh không lồi (10 thanh: 5060.85 và 5076.7 lb). Thêm multistart SLSQP (kết cấu ≤ 24 phần tử, ngân sách 8 s) ⇒ 5060.85 lb.
### Còn lại
- Kết cấu > 24 phần tử (hoặc dầm lưới mịn) vẫn là điểm dừng cục bộ của FSD/OC + KKT; "Sai số KKT" chỉ tính nhân tử của ràng buộc chuyển vị nên có thể > 0 khi ràng buộc bền đang hoạt động.
- Chiều dài tính toán Euler = k_eff × chiều dài thanh (không phân tích ổn định khung tổng thể, không P-Δ).

## Chạy trên Windows
Bấm đúp `Dop_Tool.bat` (lần đầu tự tạo .venv312 với Python 3.12 và cài thư viện) → mở Đớp Tool trong cửa sổ riêng (pywebview, nếu không có thì Edge/Chrome chế độ app). `Dop_Tool_Browser.bat` mở bằng trình duyệt thường. Đặt biến `DOP_PORT` để cố định cổng.
