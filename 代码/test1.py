import re

# 硬编码一小段文本测试
test_text = """THE PENICILLIN INQUIRY,STRANDREW DUNCAN.Minister of Supply, 
stated in reply to MAJOR LYONS (Leicester, E., U.), 
that the object of the conimittee set up to report oni penicillin was to 
sectire that all available information regarding clinical and chelimical trials and 
methods of production was coilected and exchaniged. and that everything possible was done 
to promote the most rapid development. The comim1ittee consisted- of Mr. Arthtir Mortimer.
 Dcpttty Director of Medical Sttpplies. Ministry of Suipply (chairmani: and IProfessor- A. 
 I-leming. St. Mary's llospital: IPro-, fessor ti. W. Florey, School ol' Pathology, University
  ol' Oxford: Professor Fl. Raistrick, London School of Hlygiene: Sir Robert Robinison, the
   Dyson Pcrrinis Laboratorv. Universitv of Oxford: Dr. C. R. Harribgton. Medical Research 
   Council: Dr. A. :N. Drury. Medical Research Council: Dr. V. D. Allison. Ministry of llealth; 
   Professor l: M. Flcilbron. Universitv of Cambridgc: and Licuttenant-Coloncl Sir Russell 
   Wilkinson. Nlilitary Medical Adviser. Ministrv of Supply; together with representatives of
    firns engaged in production of penicillin.."""

# 简单清洗
cleaned = re.sub(r'\s+', ' ', test_text)
cleaned = cleaned.strip()

print("清洗后:")
print(cleaned)
