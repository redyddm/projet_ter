import difflib
from recommandation_de_livres.iads.text_cleaning import nettoyage_texte

text1=nettoyage_texte("Honor and Betrayal : The Untold Story of the Navy Seals Who Captured the \"Butcher of Fallujah\"--And the Shameful Ordeal They Later Endured")
text2=nettoyage_texte("Honor and Betrayal : The Untold Story of the Navy Seals Who Captured the \"Butcher of Fallujah\"--And the Shameful Ordeal They Later Endured")

print(text1)
print(text2)

text = "Soldier Five : The Real Truth About The Bravo Two Zero Mission"

"""Top 5 recommandations pour : Soldier Five : The Real Truth About The Bravo Two Zero Mission
Show Ring Success : A Rider's Guide to Winning Strategies - Kathleen Obenland

Unmarked : Sean's Story - Margreet Asselbergs Missy Borucki Eric David Battershell

Dead by Morning - James Price Kayla Krantz

The Foundling School for Girls : She may be an orphan but she has hope for the future - Elizabeth Gill

My Bucket's Got a Hole in It - Teresa Stutso Jewell"""