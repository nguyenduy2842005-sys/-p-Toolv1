from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION

out = r"F:\07. tai lieu nghien cuu\Đớp tool\Phiên bản cuối cùng\dop_tool_v2\dop_tool_v2\output\Kich_ban_thuyet_trinh_Dop_Tool_10_phut.docx"
doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(0.72); sec.bottom_margin=Inches(0.72); sec.left_margin=Inches(0.85); sec.right_margin=Inches(0.85)
styles=doc.styles
normal=styles['Normal']; normal.font.name='Aptos'; normal.font.size=Pt(11.5); normal.font.color.rgb=RGBColor(35,45,55)
normal.paragraph_format.space_after=Pt(7); normal.paragraph_format.line_spacing=1.13
for sn,size in [('Title',25),('Heading 1',16),('Heading 2',12)]:
    s=styles[sn]; s.font.name='Aptos Display' if sn=='Title' else 'Aptos'; s.font.size=Pt(size); s.font.bold=True; s.font.color.rgb=RGBColor(0,0,0)
    s.paragraph_format.keep_with_next=True
styles['Heading 1'].paragraph_format.space_before=Pt(14); styles['Heading 1'].paragraph_format.space_after=Pt(5)
styles['Heading 2'].paragraph_format.space_before=Pt(8); styles['Heading 2'].paragraph_format.space_after=Pt(3)

p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Kịch bản thuyết trình Đớp Tool')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Ứng dụng OpenSees trong phân tích và tối ưu hóa công trình chịu động đất'); r.bold=True; r.font.size=Pt(13); r.font.color.rgb=RGBColor(40,75,60)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Thời lượng dự kiến: 10 phút  |  Đối tượng: giảng viên bộ môn và sinh viên'); r.italic=True; r.font.size=Pt(10); r.font.color.rgb=RGBColor(95,105,100)

doc.add_heading('Cách sử dụng kịch bản', level=1)
doc.add_paragraph('Kịch bản đi theo thứ tự 11 slide của file thuyết trình. Mốc thời gian là gợi ý, đã tính các khoảng dừng ngắn để chuyển slide hoặc chỉ vào hình. Có thể thay “nhóm em” bằng cách xưng hô phù hợp với người trình bày.')

slides=[
('Slide 1 — Giới thiệu đề tài','0:35','Kính thưa thầy và các bạn, nhóm em xin trình bày đề tài “Ứng dụng OpenSees trong phân tích và tối ưu hóa công trình chịu động đất”.\n\nSản phẩm nhóm đang phát triển có tên là Đớp Tool. Tool hướng đến việc hỗ trợ thiết lập mô hình kết cấu, thực hiện phân tích động đất bằng ba phương pháp ELF, MRSA và THA, sau đó trình bày kết quả để người dùng dễ theo dõi.\n\nTrong phần trình bày này, nhóm em sẽ giới thiệu bài toán, cách tổ chức chương trình và cơ sở của các thuật toán chính.'),
('Slide 2 — Bài toán nghiên cứu và phạm vi','0:55','Động đất tạo ra tác động thay đổi theo thời gian. Vì vậy, người thiết kế cần đánh giá được lực, chuyển vị và mức độ biến dạng của công trình trong nhiều điều kiện khác nhau.\n\nMục tiêu của nhóm là xây dựng một giao diện giúp người dùng thiết lập mô hình, lựa chọn phương pháp phân tích và xem kết quả trên cùng một công cụ. Nhóm cũng định hướng dùng kết quả phân tích để khảo sát phương án kết cấu nhẹ hơn, đồng thời kiểm soát các giới hạn như chuyển vị đỉnh và lệch tầng.\n\nPhạm vi hiện tại tập trung vào quy trình mô hình hóa và phân tích. Kết quả của tool cần được đối chiếu với mô hình kiểm chứng và tiêu chuẩn áp dụng trước khi dùng cho một công trình thực tế. Đây là sản phẩm học tập và nghiên cứu đang phát triển, không thay thế việc thẩm tra thiết kế.'),
('Slide 3 — Mô hình vật liệu','0:55','Đối với phân tích phi tuyến, mô hình vật liệu quyết định cách cấu kiện phản ứng khi tải trọng làm vật liệu vượt khỏi vùng đàn hồi.\n\nTrong phần bê tông, tool sử dụng Concrete01. Đây là mô hình bê tông đơn giản hóa quan hệ ứng suất và biến dạng khi chịu nén. Mô hình thể hiện nhánh tăng đến cường độ đỉnh, sau đó suy giảm. Trong cấu hình này, bê tông không chịu kéo.\n\nVới cốt thép, tool sử dụng Steel02. Mô hình này mô tả thép đàn hồi, chảy dẻo, hóa bền và ứng xử khi tải đảo chiều. Hai mô hình tạo nền tảng để tiết diện fiber có thể xét riêng phần bê tông và thép.\n\nCác tham số vật liệu cần được chọn dựa trên hồ sơ thiết kế, tiêu chuẩn hoặc thí nghiệm. Chọn đúng tên vật liệu thôi chưa đủ để đảm bảo mô hình phản ánh đúng công trình.'),
('Slide 4 — Ba phương pháp phân tích','1:10','Tool tổ chức ba phương pháp phân tích động đất.\n\nThứ nhất là ELF, Equivalent Lateral Force, hay phương pháp lực ngang tương đương. Phương pháp quy đổi tác động động đất thành hệ lực ngang tĩnh. Cách làm này tương đối gọn và phù hợp để khảo sát nhanh, nhưng có thể không thể hiện đầy đủ đóng góp của nhiều dạng dao động.\n\nThứ hai là MRSA, Modal Response Spectrum Analysis, hay phân tích phổ phản ứng dạng dao động. Phương pháp này tính phản ứng ở nhiều dạng dao động, sau đó tổ hợp các phản ứng modal. CQC hoặc SRSS được dùng tùy đặc điểm tương quan giữa các dạng dao động. MRSA cho biết ảnh hưởng của nhiều mode nhưng không mô tả diễn biến đáp ứng theo thời gian.\n\nThứ ba là THA, Time History Analysis, hay phân tích lịch sử thời gian. Phương pháp đưa gia tốc nền vào mô hình theo từng bước thời gian và tính đáp ứng tương ứng. THA có thể xét phi tuyến, nhưng kết quả phụ thuộc đáng kể vào bản ghi, bước thời gian, cách hiệu chỉnh tín hiệu và mô hình vật liệu.'),
('Slide 5 — Kiến trúc và luồng xử lý','0:55','Luồng xử lý của tool bắt đầu từ dữ liệu đầu vào: hình học, số tầng, nhịp, tiết diện, khối lượng, vật liệu và tham số động đất.\n\nTừ đó, chương trình dựng mô hình phần tử hữu hạn và thiết lập các ma trận hoặc đặc trưng cần cho phân tích. Người dùng chọn ELF, MRSA hoặc THA. Bộ giải thực hiện phân tích, sau đó tool thu các đại lượng đầu ra như chuyển vị, lệch tầng và lực cắt đáy để trình bày trên giao diện.\n\nĐiểm quan trọng trong kiến trúc là giữ mối liên hệ giữa đầu vào, mô hình tính và kết quả. Khi người dùng thay đổi một tham số, cần chạy lại phân tích để kết quả được đồng bộ với cấu hình mới.'),
('Slide 6 — Tài liệu tham khảo','0:25','Slide này tổng hợp các tài liệu nhóm dùng làm cơ sở tham khảo. Khi phát triển một công cụ phân tích kết cấu, nhóm cần dựa đồng thời vào tài liệu OpenSees, tài liệu về động lực học kết cấu và tiêu chuẩn thiết kế phù hợp.\n\nCác công thức trong tool cần được kiểm tra với nguồn tương ứng, đặc biệt là các giả thiết về phổ, tải trọng, tổ hợp dạng dao động và mô hình vật liệu.'),
('Slide 7 — Mã nguồn','0:35','Đây là phần giới thiệu cấu trúc mã nguồn. Nhóm chia chương trình thành các phần đảm nhiệm giao diện, thiết lập mô hình, phương pháp phân tích và hiển thị kết quả.\n\nCách phân chia này giúp dễ xác định vị trí cần chỉnh sửa khi phát hiện lỗi hoặc khi bổ sung phương pháp mới. Tuy vậy, việc chia module chỉ giúp tổ chức phần mềm; độ tin cậy của kết quả vẫn phải được đánh giá bằng kiểm tra số học, đối chiếu bài mẫu và kiểm chứng với phần mềm hoặc mô hình độc lập.'),
('Slide 8 — Mô hình FEM và đặc trưng dao động','1:00','Trước khi phân tích động đất, cần xác định mô hình kết cấu và đặc trưng dao động của nó.\n\nTrên slide, nhóm mô tả khung ba chiều gồm nút và phần tử dầm cột. Chân cột được khai báo ngàm theo giả thiết của mô hình. Các nút sàn cùng tầng được liên kết bằng diaphragm cứng để biểu diễn chuyển động sàn trong mặt phẳng.\n\nKhối lượng được tập hợp thành ma trận M, còn độ cứng của kết cấu được mô hình FEM biểu diễn trong ma trận K. Bài toán trị riêng có dạng Kφᵣ = ωᵣ²Mφᵣ. Kết quả cho tần số góc, chu kỳ và dạng dao động của từng mode.\n\nCác đại lượng này phụ thuộc vào hình học, vật liệu, khối lượng và điều kiện biên. Vì thế, nếu đầu vào chưa đại diện đúng công trình thì chu kỳ và đáp ứng tính được cũng sẽ bị ảnh hưởng.'),
('Slide 9 — ELF và MRSA','1:10','Với ELF, chương trình xác định chu kỳ cơ bản T₁, lấy gia tốc phổ tương ứng Sa(T₁), rồi quy đổi thành lực cắt đáy và phân bố lực lên các tầng theo mô hình sử dụng. Từ đó có thể tính các đáp ứng đàn hồi như chuyển vị và lực cắt đáy.\n\nVới MRSA, chương trình tính nhiều chu kỳ và dạng dao động. Mỗi mode có một đáp ứng riêng theo phổ thiết kế. Các đáp ứng này cần được tổ hợp vì ta không biết các cực trị modal xảy ra đồng thời theo cùng một dấu.\n\nSRSS lấy căn bậc hai của tổng bình phương các đáp ứng modal, thường phù hợp khi các mode ít tương quan. CQC đưa hệ số tương quan giữa các cặp mode vào phép tổ hợp, hữu ích hơn khi tần số các mode nằm gần nhau.\n\nTrong giao diện, nhóm đặt ngưỡng khối lượng tham gia tích lũy để quyết định số mode được giữ. Ngưỡng đang trình bày trên slide là 90 phần trăm. Khi áp dụng cho bài toán thực tế, cần kiểm tra yêu cầu tiêu chuẩn và đảm bảo số mode đủ cho từng phương phân tích.'),
('Slide 10 — THA và đáp ứng theo thời gian','1:20','THA bắt đầu bằng bản ghi gia tốc nền ag(t). Gia tốc được áp vào mô hình theo từng bước thời gian, ở đây trình bày bằng kích động nền UniformExcitation.\n\nỞ mỗi bước, chương trình giải phương trình chuyển động. Vế trái gồm ảnh hưởng của khối lượng, cản và lực kháng của kết cấu; vế phải là tác động của chuyển động nền. Với mô hình phi tuyến, lực kháng phụ thuộc vào trạng thái biến dạng và lịch sử tải trước đó.\n\nSlide mô tả tích phân Newmark với γ bằng 0,5 và β bằng 0,25. Khi phân tích phi tuyến, bộ giải dùng lặp Newton để tìm trạng thái cân bằng tại từng bước. Sau đó chương trình ghi lại đáp ứng theo thời gian như chuyển vị, lệch tầng, gia tốc hoặc lực cắt đáy.\n\nMột điểm nhóm muốn nhấn mạnh là bản ghi tạo sẵn chỉ phục vụ chạy thử giao diện. Muốn đánh giá thiết kế, người dùng cần kiểm tra nguồn bản ghi, đơn vị, bước thời gian và quy trình scale hoặc hiệu chỉnh phổ. Không nên xem mọi tín hiệu đầu vào là một trận động đất thiết kế phù hợp.'),
('Slide 11 — Tiết diện fiber và ảnh hưởng cốt thép','1:00','Trong mô hình fiber, tiết diện dầm hoặc cột được chia thành các thớ bê tông và thép. Mỗi thớ có quan hệ ứng suất–biến dạng riêng. Khi cấu kiện chịu uốn hoặc lực dọc, biến dạng trên tiết diện làm cho từng thớ phát triển ứng suất tương ứng.\n\nVới Concrete01, nhóm mô tả ứng xử bê tông chịu nén theo các tham số cường độ và biến dạng đặc trưng. Với Steel02, nhóm mô tả ứng xử của thép theo giới hạn chảy, độ cứng đàn hồi, hóa bền và các tham số chuyển tiếp khi đảo chiều tải.\n\nTọa độ và diện tích các thanh thép ảnh hưởng đến sức kháng và độ cứng của tiết diện. Do đó, thay đổi bố trí thép trong mô hình phải được đưa vào phần khai báo fiber thì phân tích mới phản ánh thay đổi đó.\n\nMô hình hiện có những giới hạn cần lưu ý, gồm chưa xét đầy đủ hiệu ứng bó hông bê tông theo Mander, tuột neo, phá hoại cắt, nứt kéo bê tông và hiệu ứng P–Delta. Đây là các nội dung nhóm cần tiếp tục bổ sung hoặc kiểm chứng.'),
('Kết thúc','0:20','Tóm lại, Đớp Tool kết nối mô hình kết cấu với ba hướng phân tích ELF, MRSA và THA trên một giao diện chung. Nhóm đang tập trung làm rõ luồng tính toán, cách nhập dữ liệu và cách diễn giải kết quả để hỗ trợ học tập và khảo sát phương án.\n\nNhóm em mong nhận được góp ý của thầy và các bạn về giả thiết mô hình, độ tin cậy của các phương pháp và những nội dung nên ưu tiên phát triển tiếp theo. Em xin cảm ơn thầy và các bạn đã lắng nghe.')]

# pacing table

doc.add_heading('Phân bổ thời lượng', level=1)
t=doc.add_table(rows=1, cols=3); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
for c,txt in zip(t.rows[0].cells,['Phần trình bày','Thời lượng','Mốc tích lũy']):
    c.text=txt; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    tcPr=c._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),'E7ECE9'); tcPr.append(shd)
for i,(title,dur,_) in enumerate(slides):
    mins,secs=map(int,dur.split(':')); total=mins*60+secs
    accum=sum(int(a.split(':')[0])*60+int(a.split(':')[1]) for _,a,_ in slides[:i+1])
    row=t.add_row().cells; row[0].text=title; row[1].text=dur; row[2].text=f'{accum//60}:{accum%60:02d}'
    for c in row: c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

for title,dur,text in slides:
    doc.add_heading(f'{title}  |  {dur}', level=1)
    for para in text.split('\n\n'):
        p=doc.add_paragraph(para)
        p.paragraph_format.keep_together=True

# footer with page field
for s in doc.sections:
    f=s.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.CENTER
    rr=f.add_run('Kịch bản giới thiệu Đớp Tool  ·  '); rr.font.size=Pt(9); rr.font.color.rgb=RGBColor(100,110,105)
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); f._p.append(fld)

doc.core_properties.title='Kịch bản thuyết trình Đớp Tool trong 10 phút'
doc.core_properties.subject='Ứng dụng OpenSees trong phân tích và tối ưu hóa công trình chịu động đất'
doc.save(out)
print(out)
