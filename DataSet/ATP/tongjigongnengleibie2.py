import requests
import pandas as pd
import time
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class EnhancedPDBAnalyzer:
    """
    增强版PDB分析器，使用多种方法获取蛋白质分类信息
    """

    def __init__(self):
        self.cache = {}
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def extract_pdb_info(self, header_line: str) -> Tuple[str, str]:
        """提取PDB ID和链ID"""
        # 移除_SMB等后缀
        clean_header = header_line.strip()

        # 尝试匹配PDB ID模式（4个字符+下划线+单个字符）
        match = re.match(r'([1-9][a-z0-9]{3})_([a-zA-Z0-9])', clean_header)
        if match:
            return match.group(1).lower(), match.group(2).upper()

        # 备用匹配：前4个字符作为PDB ID
        if len(clean_header) >= 4:
            pdb_id = clean_header[:4].lower()
            chain_id = clean_header[5] if len(clean_header) >= 6 else 'A'
            return pdb_id, chain_id

        return None, None

    def get_pdb_info_multisource(self, pdb_id: str) -> Dict:
        """
        使用多种方法获取PDB信息
        """
        if pdb_id in self.cache:
            return self.cache[pdb_id]

        info = {
            'pdb_id': pdb_id.upper(),
            'classification': 'Unknown',
            'ec_numbers': [],
            'functional_class': 'Unknown',
            'organism': 'Unknown',
            'title': '',
            'keywords': [],
            'method': 'Unknown',
            'resolution': 'N/A'
        }

        try:
            # 方法1: 使用RCSB GraphQL API（更可靠）
            info1 = self.get_pdb_info_graphql(pdb_id)
            if info1 and info1.get('classification') != 'Unknown':
                self.cache[pdb_id] = info1
                return info1

            # 方法2: 使用PDB XML接口
            info2 = self.get_pdb_info_xml(pdb_id)
            if info2 and info2.get('classification') != 'Unknown':
                self.cache[pdb_id] = info2
                return info2

            # 方法3: 使用UniProt API通过PDB ID获取信息
            info3 = self.get_uniprot_info(pdb_id)
            if info3 and info3.get('classification') != 'Unknown':
                self.cache[pdb_id] = info3
                return info3

            # 如果所有方法都失败，尝试从标题推断
            if 'title' in info1:
                info1['functional_class'] = self.infer_from_title(info1['title'])
                self.cache[pdb_id] = info1
                return info1

        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")

        self.cache[pdb_id] = info
        return info

    def get_pdb_info_graphql(self, pdb_id: str) -> Dict:
        """使用RCSB GraphQL API获取信息"""
        try:
            url = "https://data.rcsb.org/graphql"

            query = """
            query {
              entry(entry_id: "%s") {
                rcsb_id
                struct {
                  title
                  pdbx_descriptor
                }
                exptl {
                  method
                }
                refine {
                  ls_d_res_high
                }
                rcsb_entry_info {
                  resolution_combined
                }
                rcsb_primary_citation {
                  title
                }
                polymer_entities {
                  rcsb_polymer_entity {
                    pdbx_description
                    rcsb_source_organism {
                      ncbi_scientific_name
                    }
                    rcsb_ec {
                      ec
                    }
                    rcsb_enzyme_class {
                      ec
                    }
                    rcsb_cluster_membership {
                      identity
                    }
                  }
                }
                nonpolymer_entities {
                  rcsb_nonpolymer_entity {
                    pdbx_description
                    chem_comp {
                      id
                      name
                    }
                  }
                }
              }
            }
            """ % pdb_id.upper()

            response = requests.post(url, json={'query': query}, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                entry = data.get('data', {}).get('entry', {})

                if entry:
                    # 提取分类信息
                    classification = entry.get('struct', {}).get('pdbx_descriptor', 'Unknown')

                    # 获取EC编号
                    ec_numbers = []
                    polymer_entities = entry.get('polymer_entities', [])
                    for entity in polymer_entities:
                        ec_info = entity.get('rcsb_polymer_entity', {})
                        ec_list = ec_info.get('rcsb_ec', [])
                        if ec_list:
                            for ec in ec_list:
                                if 'ec' in ec:
                                    ec_numbers.append(ec['ec'])

                    # 获取物种信息
                    organisms = []
                    for entity in polymer_entities:
                        source = entity.get('rcsb_polymer_entity', {}).get('rcsb_source_organism', [])
                        if source:
                            for org in source:
                                name = org.get('ncbi_scientific_name')
                                if name and name not in organisms:
                                    organisms.append(name)

                    # 获取标题
                    title = entry.get('struct', {}).get('title', '')

                    # 获取分辨率
                    resolution = entry.get('rcsb_entry_info', {}).get('resolution_combined', ['N/A'])[0]

                    # 确定功能类别
                    functional_class = self.determine_functional_class_advanced(
                        classification, ec_numbers, title
                    )

                    return {
                        'pdb_id': pdb_id.upper(),
                        'classification': classification if classification else 'Unknown',
                        'ec_numbers': ec_numbers,
                        'functional_class': functional_class,
                        'organism': ', '.join(organisms) if organisms else 'Unknown',
                        'title': title,
                        'resolution': resolution,
                        'method': entry.get('exptl', [{}])[0].get('method', 'Unknown') if entry.get(
                            'exptl') else 'Unknown'
                    }

        except Exception as e:
            print(f"GraphQL error for {pdb_id}: {e}")

        return None

    def get_pdb_info_xml(self, pdb_id: str) -> Dict:
        """使用PDB XML接口获取信息（备用方法）"""
        try:
            url = f"https://files.rcsb.org/view/{pdb_id.upper()}.xml"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                root = ET.fromstring(response.content)

                # 命名空间处理
                ns = {'pdb': 'http://pdbml.pdb.org/schema/pdbx-v50.xsd'}

                # 提取分类
                classification_elem = root.find('.//pdb:struct_keywords/pdb:pdbx_keywords', ns)
                classification = classification_elem.text if classification_elem is not None else 'Unknown'

                # 提取标题
                title_elem = root.find('.//pdb:struct/pdb:title', ns)
                title = title_elem.text if title_elem is not None else ''

                # 提取EC编号
                ec_numbers = []
                for ec_elem in root.findall('.//pdb:pdbx_entity_src_gen/pdb:pdbx_gene_src_gene', ns):
                    if ec_elem is not None and ec_elem.text:
                        ec_numbers.append(ec_elem.text)

                # 提取物种
                organism_elem = root.find('.//pdb:entity_src_gen/pdb:pdbx_gene_src_scientific_name', ns)
                organism = organism_elem.text if organism_elem is not None else 'Unknown'

                functional_class = self.determine_functional_class_advanced(
                    classification, ec_numbers, title
                )

                return {
                    'pdb_id': pdb_id.upper(),
                    'classification': classification,
                    'ec_numbers': ec_numbers,
                    'functional_class': functional_class,
                    'organism': organism,
                    'title': title,
                    'resolution': 'N/A',
                    'method': 'Unknown'
                }

        except Exception as e:
            print(f"XML error for {pdb_id}: {e}")

        return None

    def get_uniprot_info(self, pdb_id: str) -> Dict:
        """通过UniProt API获取信息"""
        try:
            # 首先从PDB获取UniProt ID
            pdb_to_uniprot_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
            response = requests.get(pdb_to_uniprot_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                uniprot_ids = list(data.get(pdb_id.lower(), {}).get('UniProt', {}).keys())

                if uniprot_ids:
                    uniprot_id = uniprot_ids[0]

                    # 获取UniProt信息
                    uniprot_url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}"
                    uniprot_response = requests.get(uniprot_url, headers=self.headers, timeout=10)

                    if uniprot_response.status_code == 200:
                        uniprot_data = uniprot_response.json()

                        # 提取蛋白质名称和功能
                        protein_name = uniprot_data.get('protein', {}).get('recommendedName', {}).get('fullName',
                                                                                                      {}).get('value',
                                                                                                              '')

                        # 提取EC编号
                        ec_numbers = []
                        for feature in uniprot_data.get('features', []):
                            if feature.get('type') == 'ACT_SITE':
                                if 'description' in feature:
                                    desc = feature['description']
                                    # 提取EC编号
                                    ec_matches = re.findall(r'EC\s*(\d+\.\d+\.\d+\.\d+)', desc)
                                    ec_numbers.extend(ec_matches)

                        # 从UniProt分类推断功能类别
                        functional_class = self.infer_from_uniprot(uniprot_data)

                        return {
                            'pdb_id': pdb_id.upper(),
                            'classification': protein_name,
                            'ec_numbers': ec_numbers,
                            'functional_class': functional_class,
                            'organism': uniprot_data.get('organism', {}).get('names', [{}])[0].get('value', 'Unknown'),
                            'title': protein_name,
                            'resolution': 'N/A',
                            'method': 'Unknown'
                        }

        except Exception as e:
            print(f"UniProt error for {pdb_id}: {e}")

        return None

    def determine_functional_class_advanced(self, classification: str, ec_numbers: List[str], title: str) -> str:
        """高级功能类别判断"""
        text_to_analyze = (classification + ' ' + title).lower()

        # 关键词到功能类别的映射
        keyword_map = {
            # 酶类
            'hydrolase': 'Hydrolase',
            'protease': 'Protease',
            'peptidase': 'Protease',
            'lipase': 'Hydrolase',
            'nuclease': 'Hydrolase',
            'phosphatase': 'Hydrolase',
            'esterase': 'Hydrolase',
            'glucosidase': 'Hydrolase',
            'amylase': 'Hydrolase',
            'lysozyme': 'Hydrolase',

            'oxidoreductase': 'Oxidoreductase',
            'dehydrogenase': 'Oxidoreductase',
            'reductase': 'Oxidoreductase',
            'oxidase': 'Oxidoreductase',
            'peroxidase': 'Oxidoreductase',
            'oxygenase': 'Oxidoreductase',

            'transferase': 'Transferase',
            'kinase': 'Kinase',
            'methyltransferase': 'Transferase',
            'glycosyltransferase': 'Transferase',

            'lyase': 'Lyase',
            'decarboxylase': 'Lyase',
            'aldolase': 'Lyase',

            'isomerase': 'Isomerase',
            'racemase': 'Isomerase',
            'epimerase': 'Isomerase',

            'ligase': 'Ligase',
            'synthase': 'Ligase',
            'synthetase': 'Ligase',

            # 结合蛋白类
            'antibody': 'Antibody',
            'immunoglobulin': 'Antibody',
            'fab': 'Antibody',
            'fc': 'Antibody',

            'receptor': 'Receptor',
            'g-protein': 'Receptor',
            'gpcr': 'Receptor',

            'transporter': 'Transporter',
            'channel': 'Transporter',
            'carrier': 'Transporter',
            'porin': 'Transporter',
            'pump': 'Transporter',

            'dna-binding': 'Nucleic acid binding',
            'rna-binding': 'Nucleic acid binding',
            'transcription factor': 'Nucleic acid binding',
            'polymerase': 'Nucleic acid binding',
            'helicase': 'Nucleic acid binding',
            'nuclease': 'Nucleic acid binding',

            'chaperone': 'Chaperone',
            'heat shock': 'Chaperone',

            'cytokine': 'Signaling protein',
            'growth factor': 'Signaling protein',
            'hormone': 'Signaling protein',
            'signal': 'Signaling protein',

            'structural': 'Structural protein',
            'actin': 'Structural protein',
            'tubulin': 'Structural protein',
            'collagen': 'Structural protein',
            'keratin': 'Structural protein',

            'storage': 'Storage protein',
            'albumin': 'Storage protein',
            'casein': 'Storage protein',

            'motor': 'Motor protein',
            'myosin': 'Motor protein',
            'dynein': 'Motor protein',
            'kinesin': 'Motor protein',

            'toxin': 'Toxin',
            'venom': 'Toxin',

            'enzyme': 'Enzyme (general)',
        }

        # 根据EC编号判断
        if ec_numbers:
            ec_first = ec_numbers[0][0] if ec_numbers[0] else ''
            ec_map = {
                '1': 'Oxidoreductase',
                '2': 'Transferase',
                '3': 'Hydrolase',
                '4': 'Lyase',
                '5': 'Isomerase',
                '6': 'Ligase'
            }
            if ec_first in ec_map:
                return ec_map[ec_first]

        # 根据关键词判断
        for keyword, func_class in keyword_map.items():
            if keyword in text_to_analyze:
                return func_class

        # 如果还是无法判断，尝试从标题推断
        inferred = self.infer_from_title(text_to_analyze)
        if inferred != 'Other binding protein':
            return inferred

        return 'Other binding protein'

    def infer_from_title(self, title: str) -> str:
        """从标题推断功能类别"""
        title_lower = title.lower()

        # 常见蛋白质类型的关键词
        if any(word in title_lower for word in ['hemoglobin', 'myoglobin', 'globin']):
            return 'Oxygen transport'
        elif any(word in title_lower for word in ['insulin', 'glucagon', 'leptin']):
            return 'Hormone'
        elif any(word in title_lower for word in ['trypsin', 'chymotrypsin', 'elastase']):
            return 'Protease'
        elif any(word in title_lower for word in ['lysozyme', 'muranidase']):
            return 'Hydrolase'
        elif any(word in title_lower for word in ['ubiquitin', 'sumo']):
            return 'Protein modifier'
        elif any(word in title_lower for word in ['ferritin', 'transferrin']):
            return 'Iron storage/transport'
        elif any(word in title_lower for word in ['calmodulin', 'troponin']):
            return 'Calcium binding'
        elif any(word in title_lower for word in ['histone', 'nucleosome']):
            return 'Chromatin protein'

        return 'Other binding protein'

    def infer_from_uniprot(self, uniprot_data: Dict) -> str:
        """从UniProt数据推断功能类别"""
        try:
            # 检查蛋白质存在类型
            protein_existence = uniprot_data.get('proteinExistence', '')
            if 'enzyme' in protein_existence.lower():
                return 'Enzyme (general)'

            # 检查关键词
            keywords = uniprot_data.get('keywords', [])
            for keyword in keywords:
                kw_text = keyword.get('value', '').lower()
                if 'hydrolase' in kw_text:
                    return 'Hydrolase'
                elif 'transferase' in kw_text:
                    return 'Transferase'
                elif 'oxidoreductase' in kw_text:
                    return 'Oxidoreductase'

        except:
            pass

        return 'Other binding protein'

    def analyze_dataset(self, filename: str, max_workers: int = 3) -> pd.DataFrame:
        """分析整个数据集"""
        print(f"Reading data from {filename}...")

        # 读取文件并提取PDB ID
        pdb_chains = []
        with open(filename, 'r') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 2):
            if i < len(lines):
                header_line = lines[i].strip()
                pdb_id, chain_id = self.extract_pdb_info(header_line)
                if pdb_id:
                    pdb_chains.append((pdb_id, chain_id))

        print(f"Found {len(pdb_chains)} PDB chains")

        # 去重
        unique_chains = list(set(pdb_chains))
        print(f"Unique PDB chains: {len(unique_chains)}")

        # 并行获取信息
        print("Fetching PDB information from multiple sources...")
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdb = {
                executor.submit(self.get_pdb_info_multisource, pdb_id): (pdb_id, chain_id)
                for pdb_id, chain_id in unique_chains[:355]  # 限制数量避免请求过多
            }

            for i, future in enumerate(as_completed(future_to_pdb)):
                pdb_id, chain_id = future_to_pdb[future]
                try:
                    info = future.result()
                    info['chain_id'] = chain_id
                    results.append(info)

                    if (i + 1) % 5 == 0:
                        print(
                            f"Processed {i + 1}/{len(future_to_pdb)}: {pdb_id.upper()}_{chain_id} - {info['functional_class']}")

                    time.sleep(0.5)  # 避免请求过快

                except Exception as e:
                    print(f"Error processing {pdb_id}: {e}")
                    # 添加默认信息
                    results.append({
                        'pdb_id': pdb_id.upper(),
                        'chain_id': chain_id,
                        'classification': 'Unknown',
                        'functional_class': 'Unknown',
                        'organism': 'Unknown',
                        'title': ''
                    })

        # 创建DataFrame
        df = pd.DataFrame(results)
        return df

    def generate_statistics_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成统计表格"""
        # 统计功能类别
        func_counts = Counter(df['functional_class'])
        total = len(df)

        stats_data = []
        for func_class, count in sorted(func_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100

            # 获取示例
            examples = df[df['functional_class'] == func_class].head(3)
            example_str = ', '.join([f"{row['pdb_id']}_{row['chain_id']}" for _, row in examples.iterrows()])

            stats_data.append({
                'Functional Class': func_class,
                'Count': count,
                'Percentage (%)': f"{percentage:.1f}",
                'Example Structures': example_str,
                'Description': self.get_class_description(func_class)
            })

        # 添加总计
        stats_data.append({
            'Functional Class': 'TOTAL',
            'Count': total,
            'Percentage (%)': '100.0',
            'Example Structures': '',
            'Description': ''
        })

        return pd.DataFrame(stats_data)

    def get_class_description(self, func_class: str) -> str:
        """获取功能类别描述"""
        descriptions = {
            'Hydrolase': 'Catalyzes hydrolysis reactions (proteases, lipases, nucleases, etc.)',
            'Protease': 'Enzymes that cleave peptide bonds',
            'Oxidoreductase': 'Catalyzes oxidation-reduction reactions',
            'Transferase': 'Transfers functional groups between molecules',
            'Kinase': 'Phosphotransferases that add phosphate groups',
            'Lyase': 'Cleaves bonds by means other than hydrolysis or oxidation',
            'Isomerase': 'Catalyzes structural rearrangements',
            'Ligase': 'Joins molecules with covalent bonds',
            'Antibody': 'Immunoglobulin proteins for antigen recognition',
            'Receptor': 'Membrane proteins for signal transduction',
            'Transporter': 'Membrane proteins for molecular transport',
            'Nucleic acid binding': 'Proteins that bind DNA or RNA',
            'Chaperone': 'Assists protein folding and assembly',
            'Structural protein': 'Provides cellular structure and support',
            'Signaling protein': 'Involved in cell signaling pathways',
            'Storage protein': 'Stores ions or small molecules',
            'Motor protein': 'Converts chemical energy to mechanical movement',
            'Toxin': 'Harmful biological molecules',
            'Oxygen transport': 'Hemoglobin, myoglobin for oxygen binding',
            'Hormone': 'Signaling molecules in endocrine system',
            'Protein modifier': 'Modifies other proteins (ubiquitin, SUMO)',
            'Iron storage/transport': 'Ferritin, transferrin for iron handling',
            'Calcium binding': 'Calmodulin and related calcium sensors',
            'Chromatin protein': 'Histones and chromatin-associated proteins',
            'Enzyme (general)': 'General catalytic activity',
            'Other binding protein': 'Various binding functions',
            'Unknown': 'Function not determined'
        }
        return descriptions.get(func_class, 'Various biological functions')


# 主程序
def main():
    analyzer = EnhancedPDBAnalyzer()
    filename = 'ATP542.txt'

    try:
        # 分析数据（限制数量）
        print("开始分析数据集...")
        df = analyzer.analyze_dataset(filename)

        if df.empty:
            print("没有获取到数据")
            return

        print(f"\n成功获取 {len(df)} 个蛋白质的信息")

        # 显示功能类别分布
        print("\n" + "=" * 80)
        print("功能类别分布统计")
        print("=" * 80)

        stats_table = analyzer.generate_statistics_table(df)
        print(stats_table.to_string(index=False))

        # 显示前20个蛋白质的详细信息
        print("\n" + "=" * 80)
        print("蛋白质详细信息（前20个）")
        print("=" * 80)

        detail_cols = ['pdb_id', 'chain_id', 'functional_class', 'classification', 'organism']
        detailed_df = df[detail_cols].head(20)
        detailed_df.columns = ['PDB ID', 'Chain', 'Functional Class', 'Classification', 'Organism']
        print(detailed_df.to_string(index=False))

        # 保存结果
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存统计表
        stats_file = f"functional_statistics_detailed_{timestamp}.csv"
        stats_table.to_csv(stats_file, index=False)
        print(f"\n统计表已保存: {stats_file}")

        # 保存详细数据
        detail_file = f"protein_details_{timestamp}.csv"
        df.to_csv(detail_file, index=False)
        print(f"详细数据已保存: {detail_file}")

        # 生成LaTeX表格
        latex_table = stats_table.to_latex(index=False,
                                           caption='Functional Classification of Proteins in the Dataset',
                                           label='tab:functional-classification')

        latex_file = f"functional_table_{timestamp}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX表格已保存: {latex_file}")

        # 输出总结
        print("\n" + "=" * 80)
        print("分析总结")
        print("=" * 80)
        print(f"总蛋白质数量: {len(df)}")
        print(f"功能类别数量: {df['functional_class'].nunique()}")
        print(f"最常见的功能类别: {df['functional_class'].mode().iloc[0]}")

        # 显示类别分布
        print("\n类别分布:")
        for func_class in df['functional_class'].unique():
            count = (df['functional_class'] == func_class).sum()
            print(f"  {func_class}: {count}个 ({count / len(df) * 100:.1f}%)")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()