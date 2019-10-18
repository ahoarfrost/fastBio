# A custom data processing pipeline, dataset, dataloader, and databunch for biological sequencing data

## Getting raw sequence data from the SRA

I suggest you download SRA files using the prefetch and fasterq-dump tool from the SRA toolkit with options similar to:

`brombergdump --split-spot --print-read-nr --outdir $YOUROUTPUTPATH $SRRTODOWNLOAD`

Note that brombergdump does the download of the .sra file from the SRA database and then runs fasterq-dump to convert it to a .fastq file. If you're downloading a ton of files, you may want to just download the .sra and run fasterq-dump on the ones you need for a batch -- this is slower, but less disk space, which may be necessary if your .sra files take up many TB of hard disk (a .fastq file can easily be 5-10x the .sra file size, and even .fasta is maybe 3-5x the .sra file size).

--split-spot splits the reads (if they are paired) into forward and reverse strands
--print-read-nr puts the read in the sequence id header (1 or 2 for forward/reverse). You need these so sequence IDs will be unique.

The default output for fasterq-dump (and brombergdump) is a fastq file. You can convert it to other filetypes after the fact if you need to (for example, fasta would take up less disk space)

## Preparing raw sequence data 

Most commonly for your input data, you're going to have several sequence files (.fastq files) that you want to use as input. 
Each individual sequence file will be for a single sample with several million individual reads. For default language modeling, a read is the unit of input.
You *could* use full genes or genomes, etc., in which case you may need to write more custom parsers to chunk up a genome sequence into input size you want...

### Key functions 

* FastqIterator 

    The function "FastqIterator" (in data.py) takes a file handle for a fastq file, and returns the header info, sequence, quality sequence, and file offset for that record. 
    The file offset enables you to read that record directly from its location in the containing file on the fly (for model input)
    The header info is parsed downstream to create the SeqRecord with the info needed for reading the file (filename and offset)
    The sequence info is used when reading in the actual data for the model batch; 
