#!/usr/bin/env python
"""Generate a comprehensive Bihar knowledge dataset for training."""

import os
import sys

DATASET = """
Bihar is a state in eastern India. It is the third most populous state in
India with a population of over 124 million people. The state capital is
Patna, which is one of the oldest continuously inhabited cities in the world.
Bihar covers an area of 94,163 square kilometers and shares borders with
Nepal to the north, Jharkhand to the south, West Bengal to the east, and
Uttar Pradesh to the west.

The geography of Bihar is dominated by the fertile Indo-Gangetic plain,
making it one of the most densely populated and agriculturally productive
regions in India. The Ganges River flows through the middle of the state
from west to east, dividing it into two halves — North Bihar and South
Bihar. The northern plains are extremely fertile but prone to devastating
floods, particularly from the Kosi River, which is known as the Sorrow
of Bihar for its history of catastrophic flooding and course changes.

Bihar is home to some of the most important rivers in India. The Ganges
is the lifeline of the state, flowing through cities like Patna, Bhagalpur,
and Hajipur. Other major rivers include the Gandak, which originates in
Nepal and joins the Gunes near Patna, the Kosi, one of the largest tributaries
of the Ganges, the Son River, which flows from the Amarkantak plateau in
Madhya Pradesh, and the Falgu, which is considered sacred and flows near
the holy city of Gaya. These rivers support irrigation, fishing, and
transportation throughout the state.

The climate of Bihar is classified as humid subtropical. Summers are very
hot with temperatures reaching 40 to 45 degrees Celsius. The monsoon season
from June to September brings heavy rainfall, which is crucial for agriculture.
Winters are mild and pleasant with temperatures dropping to 5 to 10 degrees
Celsius at night. The state receives an average annual rainfall of about
1,200 millimeters, mostly during the monsoon. Fog and cold waves are common
in the northern districts during winter.

Bihar has an extraordinarily rich and ancient history that stretches back
over three thousand years. The region was the seat of some of the most
powerful empires in Indian and world history. The Magadha Kingdom, which
emerged around the 7th century BCE in what is now southern Bihar, grew
to become one of the sixteen Mahajanapadas or great kingdoms of ancient
India. The capital of Magadha was at Rajagriha, modern day Rajgir, and
later at Pataliputra, which is present day Patna.

The Maurya Empire, founded by Chandragupta Maurya in 322 BCE, was the
first empire to unify most of the Indian subcontinent. Chandragupta, a
brilliant strategist, overthrew the Nanda dynasty and established his
capital at Pataliputra. His advisor Chanakya, also known as Kautilya,
wrote the Arthashastra, one of the most comprehensive treatises on
governance, economics, and statecraft in the ancient world. The Arthashastra
covers topics from taxation and trade to espionage and military strategy,
and remains one of the most important works of political philosophy ever
written.

Emperor Ashoka the Great, the grandson of Chandragupta Maurya, ruled from
268 to 232 BCE and is considered one of the greatest rulers in history.
After the bloody Kalinga War, Ashoka embraced Buddhism and devoted his
empire to the spread of Dhamma, or righteous living. He erected pillars
and rock edicts across his empire spreading messages of non-violence,
tolerance, and social welfare. The Ashoka Chakra, a wheel with 24 spokes,
appears on the Indian national flag and is a direct legacy of Ashoka's
Buddhist teachings. His capital at Pataliputra was one of the largest
cities in the ancient world with a population estimated at several hundred
thousand.

The Gupta Empire, which ruled from approximately 320 to 550 CE, is often
called the Golden Age of India. The Gupta period saw remarkable advances
in science, mathematics, literature, art, and philosophy. Aryabhata, the
great mathematician and astronomer, may have been associated with the
Gupta court. He calculated the value of pi, proposed that the Earth rotates
on its axis, and made advances in algebra and trigonometry. The decimal
numeral system, which is the foundation of modern mathematics, was
developed during this period. Kalidasa, the greatest Sanskrit poet and
dramatist, flourished during the Gupta era, producing masterpieces like
Shakuntala and Meghaduta.

Nalanda, located in present day Bihar, was the world's first residential
university and one of the greatest centers of learning in the ancient
world. Founded in the 5th century CE during the Gupta period, Nalanda
attracted scholars and students from across Asia, including China, Korea,
Japan, Tibet, and Southeast Asia. At its peak, Nalanda had over 10,000
students and 2,000 teachers. The university offered studies in Buddhist
philosophy, logic, grammar, medicine, and art. The Chinese pilgrim
Xuanzang visited Nalanda in the 7th century and left detailed accounts
of its magnificent libraries and intellectual life. Nalanda was destroyed
in the 12th century by Bakhtiyar Khilji, an event that marked a turning
point in the history of Indian education.

Vikramshila, another great Buddhist university, was established by Emperor
Dharmapala in the late 8th century near present day Bhagalpur in Bihar.
Vikramshila was known for its emphasis on tantric Buddhism and attracted
scholars from Tibet, China, and Southeast Asia. The university was also
destroyed during the same wave of invasions that destroyed Nalanda. These
two institutions represented the pinnacle of Buddhist learning and their
destruction was a cultural catastrophe whose effects are still felt today.

Bihar is deeply connected to the origins of both Buddhism and Jainism,
two of the world's major religions. Siddhartha Gautama, who became the
Buddha, spent significant parts of his life in Bihar. He attained
enlightenment at Bodh Gaya under the famous Peepal tree, which is still
standing today. The Mahabodhi Temple at Bodh Gaya, a UNESCO World Heritage
Site, marks the spot where the Buddha attained Nirvana. Bodh Gaya is the
most important pilgrimage site for Buddhists worldwide, attracting hundreds
of thousands of visitors each year from countries across Asia.

Rajgir, the ancient capital of Magadha, is another important Buddhist site.
The Buddha preached his first sermon at the Gridhakuta Hill, also known as
Vulture Peak, near Rajgir. The Venuvana Vihara, a bamboo grove donated to
the Buddha by the King of Magadha, was one of the first Buddhist monasteries.
Rajgir also has important Jain temples and hot springs that have been
sacred for thousands of years. The city is surrounded by hills, making
it a place of great natural beauty and spiritual significance.

Vaishali, located in present day Muzaffarpur district, is considered
the world's first republic. The Vajjian Confederacy of Vaishali was a
democratic republic that existed as early as the 6th century BCE. The
Buddha frequently visited Vaishali and delivered several important
teachings there. It was at Vaishali that the second Buddhist Council
was held in 383 BCE to resolve disputes about Buddhist teachings and
monastic discipline. Vaishali is also the birthplace of Mahavira, the
24th and last Tirthankara of Jainism, who was born in 599 BCE.

Mahavira, the founder of Jainism as a organized religion, was born in
Vaalabhi in present day Gujarat but is closely associated with Bihar
where he spent much of his life teaching and preaching. His teachings
of ahimsa (non-violence), satya (truth), and asceticism formed the
foundation of Jain philosophy. The Jain community in Bihar has preserved
many ancient temples and traditions. The town of Pawapuri, near Rajgir,
is where Mahavira attained nirvana and his body was cremated.

The medieval period of Bihar saw the rise and fall of various dynasties
and the arrival of Islamic rule. Bakhtiyar Khilji, a Turkish general,
conquered Bihar in 1193 CE and destroyed the great Buddhist universities
of Nalanda and Vikramshila. The Delhi Sultanate and later the Mughal
Empire controlled Bihar for several centuries. During the Mughal period,
Patna became an important trading center and a center of Islamic learning.
The city was known for its production of cotton textiles, indigo, and
opium.

Sher Shah Suri, one of the most remarkable rulers in Indian history,
was born in the Rohtas district of Bihar. He founded the Sur dynasty
after defeating the Mughal Emperor Humayun in 1540. Sher Shah Suri is
credited with building the Grand Trunk Road, one of the oldest and longest
roads in Asia, connecting Chittagong in Bangladesh to Kabul in Afghanistan.
He also reformed the postal system, established a network of sarais
(rest houses), and introduced the Rupiya, the predecessor of the modern
Indian rupee. His tomb at Sasaram in Bihar is a magnificent example of
Indo-Islamic architecture.

The British East India Company established control over Bihar after the
Battle of Buxar in 1764. Patna became the capital of Bengal Presidency and
later of Bihar and Orissa Province. The Champaran Satyagraha of 1917 was
a watershed moment in India's independence movement and marked the beginning
of Mahatma Gandhi's active involvement in Indian politics. Gandhi was
invited by Raj Kumar Shukla, a farmer from Champaran, to investigate the
exploitation of indigo farmers by British planters. The Champaran Satyagraha
was Gandhi's first civil disobedience movement in India and demonstrated
the power of nonviolent resistance.

The Quit India Movement of 1942 saw massive participation from the people
of Bihar. The movement launched by Gandhi at the Gowal Tank Maidan in
Mumbai on August 8, 1942 was met with widespread protests across India,
and Bihar was one of the most active centers of resistance. Several
provisional governments were established in rural areas of Bihar during
the movement. The sacrifice of countless unnamed freedom fighters from
Bihar played a crucial role in India's eventual independence in 1947.

Bihar has produced some of the most important political leaders in
India's history. Dr. Rajendra Prasad, born in Zeradei in Siwan district,
was the first President of India, serving from 1950 to 1962. He was a
key leader of the Indian independence movement and served as the President
of the Indian National Congress. His commitment to simplicity and public
service made him a beloved figure. Jayaprakash Narayan, known as Loknayak
or the People's Leader, was a freedom fighter and social reformer who
led the Total Revolution movement against corruption in the 1970s. His
movement against the Emergency declared by Prime Minister Indira Gandhi
in 1975 restored democracy in India.

Bihar's economy is primarily agricultural. The state is one of the largest
producers of rice, wheat, maize, and pulses in India. The fertile alluvial
soil deposited by the Ganges and its tributaries makes the Indo-Gangetic
plain one of the most productive agricultural regions in the world. Bihar
is the second largest producer of litchi in India after Uttar Pradesh. The
Muzaffarpur district is famous for its Shahi litchi, which has a
Geographical Indication tag. Sugarcane, tobacco, jute, and oilseeds are
other important crops.

Makhana, or fox nut, is a unique agricultural product of Bihar. The Mithila
region of Bihar, particularly around East Champaran and Darbhanga districts,
produces about 80 percent of India's makhana. Makhana is a highly nutritious
food used in both cooking and religious rituals. The makhana industry
provides livelihoods to hundreds of thousands of farmers and workers in
the region. Bihar makhana has been given a Geographical Indication tag,
recognizing its unique quality and origin.

Sattu, a roasted gram flour, is a traditional food of Bihar and an integral
part of Bihari cuisine. It is considered a superfood due to its high protein
content and cooling properties. Sattu is used to make drinks, parathas,
and ladoos. During the hot summer months, sattu drinks are a popular and
healthy way to beat the heat. The sattu tradition of Bihar reflects the
state's deep connection to simple, nutritious food made from locally
grown grains.

Litti Chokha is the signature dish of Bihar and one of the most iconic
foods in Indian cuisine. Litti is a round ball made from whole wheat
flour stuffed with sattu (roasted gram flour) mixed with spices, and
traditionally cooked over a charcoal fire. It is served with chokha,
a spicy mash made from roasted eggplant, tomatoes, and potatoes. Litti
Chokha is more than just food in Bihar — it is a symbol of Bihari identity
and pride. The dish represents the simplicity, earthiness, and resilience
of Bihari culture. Roadside litti stalls are a common sight across Bihar
and in Bihari diaspora communities across India.

Other famous Bihari foods include thekua, a sweet snack made from wheat
flour and jaggery, served during the Chhath Puja festival. Malpua, a
sweet pancake soaked in sugar syrup, is a popular dessert. Tilkut, made
from sesame seeds and sugar, is a specialty of Gaya. Anarsa, a sweet
rice flour snack, is prepared during festivals. Khaja, a flaky pastry
made from refined flour and sugar, is another traditional sweet. The
cuisine of Bihar is characterized by its use of mustard oil, panch phoron
(a five-spice blend), and an emphasis on simplicity and nutrition.

Chhath Puja is the most important and widely celebrated festival of Bihar.
It is dedicated to the Sun God and his sister Chhathi Maiya, and is
celebrated on the sixth day of the month of Kartik in the Hindu calendar,
usually in October or November. Chhath is unique to Bihar and Jharkhand
and is celebrated with great devotion and discipline. The rituals involve
four days of fasting, prayers at sunrise and sunset by the riverbank, and
offerings of arghya to the Sun God. The festival celebrates the bond
between humans and nature and emphasizes the importance of the sun as the
source of all life. During Chhath, the riverbanks and ponds of Bihar are
beautifully illuminated, creating a magical atmosphere.

Sama Chakeva is a harvest festival celebrated in the Mithila region of
Bihar, particularly in Madhubani and Darbhanga districts. The festival
marks the arrival of migratory birds and the beginning of the winter
sowing season. Young girls celebrate the festival by making colorful
clay models of birds and performing traditional songs and dances. Sama
Chakeva reflects the deep connection between the people of Mithila and
the natural world, and the festival is a celebration of community,
creativity, and seasonal change.

Madhubani painting, also known as Mithila painting, is one of the most
famous art traditions of Bihar. Originating in the Madhubani district,
this folk art is characterized by geometric patterns, vibrant colors,
and depictions of nature, mythology, and daily life. The tradition dates
back thousands of years and was traditionally practiced by women on the
walls and floors of their homes. In 1934, British colonial officer
William G. Archer discovered Madhubani paintings after an earthquake
revealed them on the walls of houses. Since then, the art form has gained
international recognition and is now practiced on paper, canvas, and
fabric. The Madhubani railway station features Madhubani paintings on
its walls, making it one of the most beautifully decorated railway
stations in India.

Bhojpuri culture is an important part of Bihar's cultural heritage. The
Bhojpuri language, spoken by millions in Bihar and eastern Uttar Pradesh,
has a rich literary and musical tradition. Bhojpuri folk songs, known
as Birha, narrate stories of love, separation, and social justice.
Chaita, a genre of devotional songs dedicated to Lord Krishna, is
popular during the spring season. The Bhojpuri film industry, though
smaller than Bollywood, has a massive audience across North India and
among the Bhojpuri diaspora worldwide.

Mithila folk art and culture represent one of the oldest continuous
cultural traditions in India. The region of Mithila, which includes
the districts of Darbhanga, Madhubani, Sitamarhi, and Samastipur,
has a distinctive identity within Bihar. The Maithili language, one
of the 22 scheduled languages of India, has a literary tradition
dating back to the 14th century poet Vidyapati, who is considered
one of the greatest poets in Indian literature. Vidyapati's songs
and poems in Maithili are celebrated for their beauty, emotion, and
spiritual depth. They are still sung in classical music concerts
across India.

The rivers of Bihar play a central role in the state's economy and
culture. The Kosi River, originating in the Himalayas of Nepal, is
one of the largest tributaries of the Ganges. Known as the Sorrow of
Bihar, the Kosi has changed its course multiple times, causing
devastating floods that have displaced millions of people. Despite
its destructive power, the Kosi also deposits fertile silt that makes
the soil extremely productive for agriculture. The Gandak River,
originating in the Mustang district of Nepal, joins the Gunes near
Patna and is an important source of irrigation. The Son River, flowing
from Amarkantak in Madhya Pradesh, passes through southern Bihar and
joins the Gunes near Patna.

Bihar's natural resources include coal, limestone, and mica. The
Jharia coalfield area was historically part of Bihar before the
creation of Jharkhand state in 2000. The state also has deposits
of bauxite and copper. The industrial sector of Bihar has faced
challenges due to poor infrastructure and law and order issues in
the past, but recent years have seen significant improvement. The
Bihar Industrial Area Development Authority has established several
industrial areas across the state to attract investment.

The modern economy of Bihar has shown remarkable growth in recent
years. The state has consistently achieved GDP growth rates above
the national average. The Patna-based economy is growing rapidly
with expansion in services, information technology, and retail.
The Bihar Industrial Area Development Authority has attracted
investment in food processing, pharmaceuticals, and manufacturing.
The state government has focused on improving infrastructure,
including roads, bridges, and power supply, to create a more
business-friendly environment.

Patna, the capital city of Bihar, is one of the oldest continuously
inhabited cities in the world. Located on the southern bank of the
Ganges, Patna has been a center of power and learning for over two
thousand years. The city was known as Pataliputra in ancient times
and served as the capital of the Maurya and Gupta Empires. The
Patna Museum, also known as the State Museum, houses a remarkable
collection of archaeological artifacts, including the Didarganj
Yakshi, a masterpiece of Mauryan sculpture. The Patna Sahib Gurdwara
is one of the holiest Sikh shrines, marking the birthplace of Guru
Gobind Singh, the tenth Sikh Guru.

The Golghar, a massive granary built by Captain John Garstin in
1786, is one of the most iconic landmarks of Patna. The structure
was built after the devastating Bengal famine of 1770 to store grain
for the city's population. The Golghar has a unique architectural
style with a spiral staircase leading to the top, offering panoramic
views of the city and the Ganges. The Gandhi Maidan, a large
historical ground in central Patna, has been the site of many
important political rallies and public gatherings, including events
during the Indian independence movement.

Buddha Smriti Park, also known as Buddha Memorial Park, was
inaugurated in 2010 on the banks of the Gunes in Patna. The park
features a stupa containing relics of the Buddha, brought from
Myanmar. The park was built to commemorate the 2,550th birth
anniversary of the Buddha and serves as a place of meditation and
peace. The park has become an important landmark of modern Patna
and a symbol of the state's Buddhist heritage.

Bihar is home to several prestigious educational institutions. The
Indian Institute of Technology Patna, established in 2008, is one
of the newest IITs but has quickly gained a reputation for academic
excellence. The National Institute of Technology Patna, formerly
known as the Bihar College of Engineering, has a history dating
back to 1886. Patna University, established in 1917, is one of the
oldest universities in Bihar and has produced many notable alumni.
The Nalanda International University, revived in 2010 near the
ancient ruins of Nalanda, aims to recreate the spirit of the
ancient university as a center for international learning and
research.

The Nalanda ruins, a UNESCO World Heritage Site, are among the most
important archaeological sites in India. The ruins include the remains
of monasteries, temples, and residential buildings that once formed
the Nalanda Mahavihara. Walking through the ruins, visitors can still
see the remains of classrooms, libraries, and meditation halls where
thousands of students once studied. The Archaeological Survey of India
has preserved and excavated the site extensively, revealing the grandeur
of this ancient center of learning.

The Mahabodhi Temple at Bodh Gaya is a UNESCO World Heritage Site and
one of the most sacred Buddhist pilgrimage sites in the world. The
temple marks the spot where the Buddha attained enlightenment under the
Bodhi Tree. The current temple structure dates back to the Gupta period
and features a towering spire that rises to 55 meters. The temple complex
includes the Bodhi Tree, the Animeshlocha Stupa, and the Jewel Walk,
a path where the Buddha is believed to have walked during meditation.
The Mahabodhi Temple attracts pilgrims from all over the world and is
the spiritual heart of Buddhism.

Vikramshila, located near Antichak in the Bhagalpur district of Bihar,
was one of the most important Buddhist universities in India. Founded
by King Dharmapala in the late 8th century, Vikramshila was known for
its rigorous academic standards and attracted scholars from across Asia.
The university was particularly renowned for its studies in Buddhist
logic, philosophy, and tantric practices. The ruins of Vikramshila,
now an archaeological site, include remains of temples, stupas, and
monasteries. The site was discovered in 1937 and has been excavated
extensively since then.

Rajgir, nestled among hills in the Gaya district, is one of the most
important historical and religious sites in Bihar. The city was the
capital of Magadha before Patna and is associated with both Buddhism
and Jainism. The Gridhakuta Hill, or Vulture Peak, where the Buddha
delivered many important sermons, offers stunning views of the surrounding
landscape. The Rajgir Hills are also famous for their hot springs,
particularly the Brahma Kund, which is considered sacred. The modern
town of Rajgir is a popular tourist destination that combines historical
significance with natural beauty.

Vaishali, located in the Muzaffarpur district, holds the distinction
of being the world's first republic. The Vajjian Confederacy that
governed Vaishali operated on principles of democracy and consensus
decision-making more than 2,500 years ago. The archaeological remains
at Vaishali include the Ananda Stupa, the Ashoka Pillar, and the
remains of ancient monasteries. The Abhishek Pushkarni, or coronation
tank, is believed to have been used for the coronation of Vajjian
leaders. Vaishali is an important pilgrimage site for both Buddhists
and Jains.

Sasaram, located in the Rohtas district, is the birthplace of Sher
Shah Suri, one of the most remarkable rulers in Indian history. The
tomb of Sher Shah Suri, built in 1545, is a magnificent example of
Indo-Islamic architecture. The tomb is situated in the middle of an
artificial lake and features a massive sandstone dome that rises to
30 meters. The tomb complex also includes the tomb of Sher Shah's
father, Hasan Khan Suri, and a mosque. The Sher Shah Suri tomb is
considered one of the finest examples of funeral architecture in India.

Bihar's transportation infrastructure has improved significantly in
recent years. The Patna Metro, currently under construction, will
be the first metro rail system in Bihar. The state has an extensive
network of national highways connecting Patna to other major cities.
The Jay Prakash Narayan International Airport in Patna is the busiest
airport in Bihar with connections to major cities across India. The
East Central Railway, headquartered in Hajipur, manages the railway
network in the state. Patna Junction is one of the busiest railway
stations in India.

The healthcare system of Bihar has been expanding with the establishment
of new medical colleges and hospitals. The All India Institute of Medical
Sciences Patna, established in 2012, is a premier medical institution.
The Patna Medical College Hospital, established in 1925, is one of the
oldest medical institutions in Bihar. The state government has launched
several health insurance schemes to provide affordable healthcare to
the rural population. The Ayushman Bharat scheme has benefited millions
of Bihari families by providing cashless treatment at empanelled hospitals.

The people of Bihar are known for their warmth, hospitality, and
resilience. The Bihari diaspora, one of the largest in India, has
made significant contributions in every field. Bihari migrants are
found in every major city of India and have excelled in business,
academics, politics, and the arts. The concept of Ganga-Jamuni Tehzeeb,
the syncretic culture that emerged from the interaction of Hindu and
Muslim communities, is deeply embedded in Bihari society. Bihari culture
values education, hard work, and community bonds.

The literary tradition of Bihar is rich and diverse. Vidyapati, who
lived in the 14th century, is considered the greatest Maithili poet.
His compositions, known as Padavali, explore themes of love, devotion,
and nature with extraordinary beauty and sensitivity. Ramdhari Singh
Dinkar, a freedom fighter and poet, is known for his powerful patriotic
poetry. His epic work Rashmirathi narrates the story of Karna from the
Mahabharata with great eloquence. Phanishwar Nath Renu, known for his
novel Maila Anchal, pioneered the progressive literary movement in
Hindi literature and captured the life and struggles of rural Bihar
with remarkable authenticity.

Modern Bihar has been experiencing a period of significant transformation.
The state government has focused on improving governance, law and order,
and infrastructure. The Saat Nischay program, launched in 2015, includes
seven commitments covering electricity, toilets, drinking water, roads,
and other basic amenities. The Bihar government has also invested heavily
in education, with initiatives to improve school infrastructure, teacher
training, and student enrollment. The economic growth rate of Bihar has
consistently been among the highest in India in recent years.

The Madhubani railway station, decorated entirely with Madhubani paintings,
has become a symbol of the state's cultural richness. The paintings on
the walls depict scenes from Hindu mythology, nature, and folk traditions,
making it one of the most visually stunning railway stations in India.
The project was undertaken to promote Madhubani art and give recognition
to the artists of the region. Similar art projects have been undertaken
at other railway stations across Bihar, transforming utilitarian spaces
into canvases for cultural expression.

Bihar's contribution to Indian politics is immeasurable. The state has
produced several prime ministers and presidents of India. The state has
been the birthplace of political movements that changed the course of
Indian history, from the Champaran Satyagraha to the JP Movement. The
political culture of Bihar is vibrant and active, with high voter
participation and passionate political engagement at every level of
society.

The agriculture sector of Bihar continues to be the backbone of the
economy, employing over 80 percent of the workforce. The state government
has introduced various schemes to modernize agriculture, including
subsidies for irrigation, fertilizers, and farm equipment. The
Bihar Agricultural University in Sabour is working on developing
improved crop varieties and farming techniques suited to local
conditions. The state is also promoting organic farming and has
established several organic farming clusters in different districts.

The sports culture of Bihar is growing rapidly. Cricket is the most
popular sport, with the Bihar Cricket Association governing the sport
at the state level. Patna has hosted several Ranji Trophy and domestic
cricket matches. Football and hockey also have a following, particularly
in rural areas. The state government has been investing in sports
infrastructure, including the construction of new stadiums and training
facilities. Bihar has produced several national level athletes who
have represented India in various international competitions.

Bihar's textile industry includes handloom and powerloom sectors. The
Tant saree, the famous cotton saree of Bengal, is also woven in parts
of Bihar. The silk industry of Bhagalpur, known as Tussar silk or
wild silk, is one of the oldest in India. Bhagalpur silk is renowned
for its quality and is exported worldwide. The handloom weavers of
Bhagalpur produce beautiful sarees, dress materials, and fabric from
Tussar silk, preserving a tradition that dates back centuries.

The folk music and dance traditions of Bihar are vibrant and diverse.
Jat-Jatin, a folk dance performed by women during the monsoon, tells
the story of a married woman missing her husband. Jhijhiya is a
folk dance performed during the Dashain festival, where women dance
with earthen pots balanced on their heads. Domkach is a folk dance
performed during weddings in the Mithila region. These folk traditions
represent the rich cultural tapestry of Bihar and are an important
part of the state's identity.

Bihar's connection to Jainism extends beyond the birthplace of Mahavira.
The state has several important Jain pilgrimage sites. The Jain temple
at Pawapuri, where Mahavira attained nirvana, is a beautiful marble
structure on an island in a lake. The Parsvanath Temple at Pawapuri
is one of the most revered Jain temples. The town of Champapuri in
Bhagalpur is another important Jain site. These sites attract Jain
pilgrims from across India and are a testament to the deep Jain
heritage of Bihar.

The Ganges, which flows through the heart of Bihar, is not just a
river but a cultural and spiritual lifeline. The ghats of Patna along
the Ganges are centers of religious activity, with daily prayers,
rituals, and festivals taking place by the river. The Patna Ghat
has been an important trading point for centuries, connecting Bihar
to the rest of India through river trade. The evening Ganges Aarti
at Patna is a spectacular spiritual event that attracts devotees
and tourists alike.

Bihar's education landscape includes several ancient and modern
institutions. The Vikramshila University ruins near Bhagalpur are
being developed as an educational hub. The Kameshwar Singh Darbhanga
Sanskrit University is one of the oldest Sanskrit universities in
India. The Lalit Narayan Mithila University in Darbhanga serves the
Mithila region. The Bhimrao Ambedkar Bihar University in Muzaffarpur
is another important institution. These universities, along with
numerous colleges and schools, contribute to Bihar's tradition of
learning that stretches back to Nalanda and Vikramshila.

The modern infrastructure of Bihar includes improved road networks,
power supply, and telecommunications. The state government has
launched the Jeevika program, a rural livelihoods project that has
empowered millions of rural women through self-help groups. The
program has been internationally recognized as a model for rural
development. The Bihar Rural Livelihoods Promotion Society has
helped millions of families above the poverty line through access
to credit, markets, and skill development.

The architectural heritage of Bihar includes a mix of ancient,
medieval, and colonial structures. The Rohtasgarh Fort, built in
the 16th century, is one of the largest hill forts in India. The
Mundeshwari Temple in Kaimur district is considered one of the
oldest functional Hindu temples in India, dating back to the
early centuries CE. The Maner Sharif, a Sufi shrine near Patna,
is a beautiful example of medieval Indo-Islamic architecture.
These structures reflect the diverse cultural and religious
traditions that have shaped Bihar over millennia.

Bihar's contribution to Indian cinema is significant. The state has
produced several acclaimed filmmakers and actors. Prakash Jha, a
filmmaker known for films like Gangaajal and Apaharan, has used
cinema to explore social and political issues in Bihar. Shatrughan
Sinha, born in Patna, became one of the most popular actors in
Bollywood during the 1970s and 1980s. The Bhojpuri film industry,
centered in Bihar and eastern Uttar Pradesh, produces hundreds of
films each year and has a massive audience across North India and
the global Bhojpuri diaspora.

The wildlife and natural heritage of Bihar include several important
protected areas. The Valmiki National Park in West Champaran district
is the only national park in Bihar and is home to tigers, leopards,
sloth bears, and numerous bird species. The Kanwar Lake in Begusarai
is the largest freshwater oxbow lake in Asia and an important bird
sanctuary. The Vikramshila Gangetic Dolphin Sanctuary in Bhagalpur
is dedicated to the protection of the endangered Ganges River dolphin.
These protected areas play a crucial role in preserving Bihar's natural
biodiversity.

The traditional medicine systems of Bihar include Ayurveda, Unani,
and folk medicine. The state has a long tradition of using herbs
and natural remedies for healthcare. The Patna Ayurveda College,
established in 1924, is one of the oldest Ayurveda institutions in
India. Many villages in Bihar still rely on traditional healers
and herbal remedies passed down through generations. The integration
of traditional and modern medicine is an important aspect of
healthcare in Bihar.

Bihar's contribution to Indian mathematics and science is ancient and
profound. Aryabhata, who may have been associated with the Gupta
court in Bihar, wrote the Aryabhatiya, a mathematical and astronomical
treatise that introduced the concept of zero, calculated the value of
pi, and proposed the rotation of the Earth on its axis. Varahamihira,
another great mathematician and astronomer from the region, made
important contributions to astrology, mathematics, and natural sciences.
These ancient scholars laid the foundation for modern mathematics and
science.

The rivers and waterways of Bihar have shaped the state's history and
culture. The Ganges, Gandak, Kosi, Son, and Falgu rivers have not
only provided water for agriculture but have also served as routes
for trade and cultural exchange. The boatmen of the Ganges have a
rich tradition of songs and stories. River festivals, fishing
communities, and riverside settlements are all part of Bihar's
riverine culture. The management of floods and water resources
remains one of the most important challenges facing the state.

Bihar's industrial heritage includes the historically important
 industries of the colonial era. The mica mining industry in the
Jhajha and Koderma regions (now in Jharkhand) was a major source
of revenue. The jute mills along the Ganges produced textiles for
export. The sugar industry, based on sugarcane grown in the fertile
plains, continues to be important. In recent years, Bihar has seen
growth in food processing, information technology, and services.

The folk traditions of Bihar include a rich collection of songs,
stories, and rituals that have been passed down through generations.
Alha and Udal, a ballad tradition from the Magadha region, narrates
the heroic deeds of two legendary warriors. Sohar songs are sung
during childbirth to celebrate the arrival of a new life. Jhijhiya
songs are sung during the autumn season. These folk traditions
represent the oral history and cultural memory of the Bihari people.

Bihar's contribution to Indian philosophy and spirituality is deep
and enduring. The state has been a center of philosophical thought
since the time of the Buddha and Mahavira. The Nalanda tradition
of Buddhist philosophy influenced thought across Asia. The Bhakti
movement found expression in Bihar through saints and poets who
composed devotional songs in local languages. The Sufi tradition
also took root in Bihar, with shrines like Maner Sharif becoming
centers of interfaith harmony.

The modern political landscape of Bihar is characterized by active
democratic participation and vibrant political parties. The state
has been at the forefront of social movements, including the Mandal
Commission movement, which reshaped Indian politics. The panchayati
raj system has been strengthened in Bihar, giving local governance
powers to elected representatives at the village level. Women's
participation in politics and governance has increased significantly
through reservation of seats in local bodies.

Bihar's natural beauty, though often overshadowed by its historical
and cultural significance, is diverse and stunning. The Rajgir Hills
offer panoramic views of the surrounding plains. The Valmiki Tiger
Reserve in West Champaran is a lush forest landscape. The Ganges
riverfront at Patna offers beautiful sunset views. The tea gardens
of North Bihar near the Nepal border are picturesque. The Madhubani
countryside, with its painted houses and lush fields, is a living
canvas of folk art and rural beauty.

The Bihari diaspora has made significant contributions to India and
the world. Bihari migrants have built businesses, careers, and
communities across India and abroad. The labor and entrepreneurial
spirit of Bihari migrants have contributed to the economic development
of many states. In recent years, there has been a growing trend of
return migration, with many Biharis coming back to invest in their
home state as Bihar's economy and infrastructure improve.

Bihar stands at a crossroads of ancient heritage and modern aspiration.
The state that gave India some of its greatest empires, universities,
and cultural traditions is now working to rebuild its economy and
infrastructure for the 21st century. With its young population,
fertile land, and rich cultural heritage, Bihar has the potential
to become one of India's most dynamic states. The spirit of resilience
and renewal that has defined Bihar for millennia continues to drive
the state's transformation today.
"""

def main():
    out_path = os.path.join(os.path.dirname(__file__), "bihar.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(DATASET.strip())
    size = os.path.getsize(out_path)
    chars = len(DATASET.strip())
    words = len(DATASET.split())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Dataset created: bihar.txt")
    print(f"  Size: {size:,} bytes ({size/1024:.1f} KB)")
    print(f"  Characters: {chars:,}")
    print(f"  Words: {words:,}")

if __name__ == "__main__":
    main()
