def FastqIterator(handle, offset=0):
    #this function is copied from FastqGeneralIterator from the BioPython pkg, except added return of offset
    #you only need to provide the 'offset' argument for the situation where you are using the built-in seek() function to skip to a particular line; if you are iterating through all lines this is unnecessary
    
    # We need to call handle.readline() at least four times per record,
    # so we'll save a property look up each time:
    handle_readline = handle.readline
    offset = offset
    next_offset = None
    line = handle_readline()
    if not line:
        return  # Premature end of file, or just empty?
    if isinstance(line[0], int):
        raise ValueError("Is this handle in binary mode not text mode?")
    while line:
        if next_offset:
            offset = next_offset
        if line[0] != "@":
            raise ValueError("Records in Fastq files should start with '@' character")
        title_line = line[1:].rstrip()
        # Will now be at least one line of quality data - in most FASTQ files
        # just one line! We therefore use string concatenation (if needed)
        # rather using than the "".join(...) trick just in case it is multiline:
        seq_string = handle_readline().rstrip()
        # There may now be more sequence lines, or the "+" quality marker line:
        while True:
            line = handle_readline()
            if not line:
                raise ValueError("End of file without quality information.")
            if line[0] == "+":
                # The title here is optional, but if present must match!
                second_title = line[1:].rstrip()
                if second_title and second_title != title_line:
                    raise ValueError("Sequence and quality captions differ.")
                break
            seq_string += line.rstrip()  # removes trailing newlines
        # This is going to slow things down a little, but assuming
        # this isn't allowed we should try and catch it here:
        if " " in seq_string or "\t" in seq_string:
            raise ValueError("Whitespace is not allowed in the sequence.")
        seq_len = len(seq_string)

        # Will now be at least one line of quality data...
        quality_string = handle_readline().rstrip()
        # There may now be more quality data, or another sequence, or EOF
        # If this is the end of the read sequence, should look at offset here; this is where the NEXT 
        next_offset = handle.tell()
        while True:
            line = handle_readline()
            if not line:
                break  # end of file
            if line[0] == "@":
                # This COULD be the start of a new sequence. However, it MAY just
                # be a line of quality data which starts with a "@" character.  We
                # should be able to check this by looking at the sequence length
                # and the amount of quality data found so far.
                if len(quality_string) >= seq_len:
                    # We expect it to be equal if this is the start of a new record.
                    # If the quality data is longer, we'll raise an error below.
                    break
                # Continue - its just some (more) quality data.
            quality_string += line.rstrip()
            #if there's more quality data, take offset after that line until we break
            next_offset = handle.tell()

        if seq_len != len(quality_string):
            raise ValueError(
                "Lengths of sequence and quality values differs for %s (%i and %i)."
                % (title_line, seq_len, len(quality_string))
            )

        # Return the record and then continue...
        yield (title_line, seq_string, quality_string, offset)