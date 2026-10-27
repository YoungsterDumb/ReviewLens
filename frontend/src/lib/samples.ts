export type SampleId = 'vi_shoe' | 'en_laptop' | 'vi_phone'

export interface Sample {
  id: SampleId
  text: string
  image?: string
}

export const SAMPLES: Sample[] = [
  {
    id: 'vi_shoe',
    text:
      'Mua đôi giày này cho chuyến đi Đà Lạt. Đế êm, đi cả ngày không đau chân. Nhưng màu ngoài đời tối hơn hình, ' +
      'và đường may ở mũi giày bị bung chỉ chỉ sau một tuần. Giao hàng nhanh, đóng gói cẩn thận. Giá hơi cao so với chất lượng.',
    image: '/samples/sneaker.png',
  },
  {
    id: 'en_laptop',
    text:
      'The screen is gorgeous and the keyboard feels great, but the battery barely lasts four hours and the fan gets loud ' +
      'under load. Customer support was helpful when I asked about the warranty.',
  },
  {
    id: 'vi_phone',
    text: [
      'Mình dùng máy được hơn một tháng rồi nên viết đánh giá chi tiết cho mọi người tham khảo.',
      'Màn hình đẹp, màu sắc tươi, xem phim rất đã, độ sáng ngoài trời cũng ổn.',
      'Pin trâu, dùng mạng xã hội và xem video cả ngày vẫn còn khoảng 30%.',
      'Sạc nhanh, khoảng một tiếng là đầy.',
      'Camera chụp ban ngày đẹp, chi tiết, nhưng chụp đêm thì bị nhoè và nhiễu khá nhiều.',
      'Hiệu năng mượt, chơi game nhẹ không giật, có điều chơi lâu máy hơi nóng ở phần trên.',
      'Loa ngoài to nhưng âm hơi rè khi để âm lượng lớn.',
      'Bộ nhớ 256GB thoải mái, cài nhiều ứng dụng vẫn còn trống.',
      'Thiết kế gọn, cầm chắc tay, mặt lưng không bám vân tay.',
      'Nhân viên cửa hàng tư vấn nhiệt tình, giao hàng đúng hẹn.',
      'Giá hơi cao so với các máy cùng phân khúc nhưng nhìn chung mình vẫn hài lòng.',
    ].join(' '),
  },
]

export const INSIGHTS_SAMPLE = [
  'Giao hàng nhanh, đóng gói cẩn thận.',
  'Đế êm, đi cả ngày không đau chân.',
  'Đường may bị bung sau một tuần, rất thất vọng.',
  'Màu ngoài đời tối hơn hình.',
  'Giá hơi cao so với chất lượng.',
  'Pin trâu nhưng camera chụp đêm bị nhoè.',
  'Màn hình đẹp, loa hơi rè.',
  'The battery barely lasts four hours.',
  'Great screen and a comfortable keyboard.',
  'Delivery was fast and the packaging was solid.',
  'The fan is loud and the laptop gets hot.',
  'Customer support was helpful.',
].join('\n')
