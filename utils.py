import numpy as np
import pandas as pd
import pymarc as pm
def get_record_metadata(record, field_no, subfields, restraint=[]):
    """
    Retrieves Information of a MARCXML record, given a field number, a list of subfields and a specific field type.

    If more than one subfield is selected, concatenates all subfield values to the returned string.

    :param record: pymarc.Record object
        The record to be searched.
    :param field_no: str
        Field of record to be checked.
    :param subfields: lst of str
        Subfields to be extracted.
    :param restraint: lst, len < 3
        List of specific fields to be checked, if list is empty gets first instance of field in record.
    :return: str
        String of retrieved information, if field_no isn't found in the record, the string will be empty.
    """
    res = ''
    if not restraint:
        # look for match on field_no in record else return empty string
        field = record.get_fields(field_no) if record.get_fields(field_no) else None
        if not field:
            return res
        else:
            # iterate over all fields with matching field no only gets first subfield instance for now
            for sub in field:
                # add metadata information of subfield
                for i, subfield in enumerate(subfields):
                    # add space if multiple subfields
                    if i > 0 and res:
                        if sub.get_subfields(subfield):
                            res += ' ' + sub.get_subfields(subfield)[0]
                    else:
                        if sub.get_subfields(subfield):
                            res += sub.get_subfields(subfield)[0]
                break
        return res
    else:
        # restraint condition if looking for specific fields like rkv or gnd does not provide support for more than one subfield
        field = record.get_fields(field_no) if record.get_fields(field_no) else None
        if not field:
            return ''
        else:
            # iterate over all subfields to find restraint condition
            for sub in field:
                if sub.get_subfields('2'):
                    if sub.get_subfields('2')[0] == restraint[0] or sub.get_subfields('2')[0] == restraint[1]:
                        if res:
                            res +=  ', ' + sub.get_subfields(subfields[0])[0]
                        else:
                            res += sub.get_subfields(subfields[0])[0]
        return res

def clean_first_word(s):
    #Regex for cleaning title statement
    return re.sub(r'^\x98(\S+)\x9c', r'\1', s)
def roman_to_int(s):
    # converts roman numerals to int
    roman_values = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000
    }
    total = 0
    prev = 0
    for c in reversed(s):
        val = roman_values[c]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total
def extract_roman_numerals(s):
    # extracts roman numerals of a string and converts to int, if none are found returns 0
    num = re.search(r'\b[IVXLCDM]+\b', s)
    if not num:
        return 0
    else:
        return roman_to_int(num.group(0))
def get_pages(s):
    # gets pages based on if number is followed by 'S.' or 'Seiten'
    s = s.replace("[", "").replace("]", "")
    match = re.search(r'(\d+)\s+(?:S\.|Seiten|Bl\.)', s)
    if match:
        return int(match.group(1))
    else:
        return 0
def get_date(s):
    match = re.search(r'\b\d{4}\b', s)
    if match:
        return(int(match.group()))
    else:
        return 0