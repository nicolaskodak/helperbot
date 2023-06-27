import re

url_pattern = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

def parse_url( answer: str) -> list:
	"""
	Argument
		--------
		answer: str
			e.g. "【  `Source`: https://draiden.org/acromioclavicular-joint-injury/  】"
	"""
	return [ x[0].replace("】","").replace("【", "") for x in url_pattern.findall(answer)]


def parse_element(answer: str, element: str):
	"""
	Arguments:
	"""
	sq_bracket_pattern = re.compile("\【[[^\【^\】^`]*``{element}``:([^\【^\】^`]+)\】")
	found = sq_bracket_pattern.findall(answer)
	parsed = None
	if found and len(found)>0:
		parsed = found[0][0].strip()
	return { element: parsed}


