from src.utils.chromosome import Chromosome

chromosomes: list[Chromosome] = []
for i in range(10):
    chrom = Chromosome(10)
    chrom.insert_crossover(i)
    chromosomes.append(chrom)

for chrom in chromosomes:
    print(chrom)
    print(chrom.has_crossover_at(0), chrom.has_crossover_at(1), chrom.has_crossover_at(2), chrom.has_crossover_at(3))
    print(chrom.has_visible_crossover(0, 3))
