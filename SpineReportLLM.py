from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("../.env")

TEMPLATE = """ Objective: This prompt analyzes a Portuguese spinal column MRI or CT report and extracts the following categories of information and their level of occurrence into a JSON object. 
If a category is not mentioned in the report, the JSON object will show "None" for that category.


Instructions:
Input: Provide the prompt with a single Portuguese spinal column MRI or CT report.
Synonym Recognition: The prompt should take into account synonyms and related terms for each category to improve accuracy and comprehensiveness.
Level of Occurrence: Identify the specific vertebral level(s) at which each finding occurs.
Multiple Occurrences: Consider multiple occurrences of the same category at different levels. If a category occurs at multiple levels, list all the levels in the JSON object. 
Some categories may be labeled as "Difuso" 

Note: Only the JSON object should be present in the output. Follow the categories strictly as mentioned in the prompt. Do not create new categories or modify the existing ones. 

Incorporate the following synonyms and related terms to enhances the accuracy and comprehensiveness of information extraction from the Portuguese spinal MRI and CT reports.


Categories (with common synonyms):
Escoliose: desvio lateral da coluna, curvatura lateral da coluna, escoliose lombar, escoliose toracolombar
Espondilose: osteofitose marginal, osteófitos, bicos de papagaio, degeneração facetária, artrose facetária
Osteófito: osteófito, osteófito marginal, bico de papagaio, reação osteofitária, reações osteofitárias
Osteófito anterior: osteófito anterior, osteófito marginal anterior, bico de papagaio anterior
Listese Grau 1: listese grau I, deslizamento vertebral grau I, escorregamento vertebral grau I, anterolistese grau I, retrolistese
Listese Grau 2 ou superior: listese grau II, listese grau III, listese grau IV, deslizamento vertebral grau II ou superior, escorregamento vertebral grau II ou superior, anterolistese grau II, anterolistese grau III, anterolistese grau IV, anterolistese grau IV
Espondilólise: lise dos istmos, lise da pars interarticularis, fratura dos istmos
Fratura: fratura vertebral, fratura compressiva, fratura impactada, fratura por insuficiência, acunhamento vertebral, colapso vertebral
Modic Tipo 1: edema da placa terminal, edema ósseo subcondral, Modic tipo I, degeneração discal edematosa
Fissura Anular: fissura do anel fibroso, rotura do anel fibroso, ruptura anular
Degeneração discal: degeneração do disco, discopatia degenerativa, disco desidratado, disco hipohidratado, desidratação discal, hipoidratação discal, perda de sinal do disco
Perda de Altura do Disco: redução da altura discal, perda da altura do disco, disco com altura reduzida
Abaulamento discal: abaulamento do disco, disco abaulado
Protusão discal: protrusão do disco, herniação discal protrusa
Extrusão discal: extrusão do disco, herniação discal extrusa, disco extruso
Degeneração Facetária: artrose facetária, osteoartrose facetária, artropatia degenerativa interfacetária, hipertrofia facetária
Estenose Central: estenose do canal vertebral, estenose do canal central, redução da amplitude do canal vertebral
Estenose Foraminal: estenose foraminal, redução da amplitude foraminal, obliteração da base foraminal
Contato com a Raiz Nervosa: contato com a raiz, proximidade com a raiz, tocando a raiz, 
Raiz Nervosa Deslocada/Comprimida: deslocamento da raiz , compressão da raiz , raiz nervosa deslocada, raiz nervosa comprimida, deslocando a raiz, comprimindo a raiz
Estenose do Recesso Lateral: estenose do recesso lateral, redução da amplitude do recesso lateral, obliteração do recesso lateral

Output: Generate a JSON object with the following structure:
{
  "Category": {
    "Listese Grau 1": ["None"],
    "Listese Grau 2 ou superior": ["None"],
    "Escoliose": ["None"],
    "Fratura": ["None"],
    // ... other categories and their corresponding levels of occurrence
  }
}


Example 1:
Input:
"RESSONÂNCIA MAGNÉTICA DA COLUNA LOMBOSSACRA\xa0 Método:\xa0 Realizadas sequências FSE ponderadas em T1 e T2. Planos de cortes múltiplos.\xa0 Análise:\xa0 Presença de megapófises transversas em L5, neoarticuladas ao sacro, notadamente à esquerda.  Escoliose toracolombar à esquerda. Anterolistese grau I de L3 e de L4, de aspecto degenerativo. Corpos vertebrais com altura e alinhamento posterior conservados, com osteófitos marginais difusos e predominantemente anteriores.  Não se observam lesões ósseas focais agressivas. Desidratação  discal  difusa  com  redução  das  alturas,  poupando  relativamente  L5-S1.  Observam-se  irregularidades dos platôs vertebrais com  degeneração  gasosa  intradiscal  e  tênue  edema (Modic tipo I) nos platôs vertebrais apostos de L4-L5.  Pequenas protrusões discais posteriores centrais à direita em T11-T12 e T12-L1, moldando o saco dural sem conflitos radiculares. Abaulamentos  discais  posteriores  em  L1-L2  a L3-L4, assimétricos, com componentes  protrusos subarticular/posterior central direito, moldando o saco  dural  e  se  insinuando  levemente  para  as  respectivas  bases  foraminais, sem conflitos radiculares definidos. Exposição  discal associada a abaulamento posterior em L4-L5, que molda o saco  dural e se insinua para as bases foraminais e que em conjunto com a listese  anterior e a hipertrofia degenerativa das interapofisárias reduz a  amplitude  foraminal  esquerda,  tocando  a raiz emergente de L4 deste lado. Abaulamento  discal  posterior  em  L5-S1  com  fissura do anel fibroso e componente  protruso  foraminal  esquerdo,  tocando  a  raiz  emergente  esquerda de L5. Alterações  degenerativas  difusas  das  articulações  interapofisárias,  predominando  inferiormente  e  notadamente  em  L4-L5 no qual se observa pequeno  derrame  articular  bilateral.  Nota-se,  também,  cisto  artrossinovial  intracanal  à esquerda de L4-L5 e medindo cerca de 0,7 cm moldando a face posterolateral esquerda do saco dural. Espessamento dos ligamentos amarelos em L2-L3 a L4-L5 Canal vertebral sem estenoses significativas. Demais forames intervertebrais livres. Cone medular tópico e com o sinal homogêneo. Cisto de Tarlov interior do canal vertebral sacral, com remodelamento associado, de aspecto crônico/sequelar.  Hipertrofia dos processos espinhosos associada a redução dos espaços interespinhosos por provável atrito crônico. Hipotrofia e substituição gordurosa da musculatura paravertebral posterior inferior. Os achados da bacia serão descritos em relatório do estudo dirigido realizado na mesma data. Volumosa  distensão  líquida  junto  em  topografia  do  endométrio,  com espessamento  irregular das paredes, parcialmente incluídos neste estudo.  Opinião: Espondilodiscoartropatia  degenerativa  multissegmentar da coluna lombar, com  as  repercussões  foraminais  e  radiculares  pormenorizadas  acima, notadamente  em  L4-L5.  Não  se  observam  alterações  evolutivas  significativas  dignas  de  nota  quando  comparado  com  o  estudo  de  tomografia  computadorizada  da  coluna lombar de 08/07/2022, respeitando as diferenças técnicas entre os métodos. Volumosa  distensão  líquida  junto  em  topografia  do  endométrio,  com espessamento  irregular das paredes, parcialmente incluídos neste estudo. Sugere-se  complementação  com  estudo  específico  para melhor avaliação deste achado, a critério clínico.  Demais achados acima descritos."

Output:

{"Category": {
  "Escoliose": "Esquerda",
  "Espondilose": ["Difusa"],
  "Osteofito": ["Difuso"],
  "Osteofito - coluna anterior": ["Difuso"],
  "Listese Grau 1": ["L3", "L4"],
  "Listese Grau 2 ou superior": ["None"],
  "Espondilólise": ["None"],
  "Fratura": ["None"],
  "Modic Tipo 1": ["L4-L5"],
  "Fissura Anular": ["L5-S1"],
  "Degeneração discal": ["Difuso"],
  "Perda de Altura do Disco": ["Difuso"],
  "Abaulamento discal": ["L1-L2","L2-L3","L3-L4", "L4-L5","L5-S1"],
  "Protrusão discal": ["T11-T12","T12-L1"],
  "Extrusão discal": ["None"],
  "Degeneração Facetária": ["Difuso"],
  "Estenose Central": ["None"],
  "Estenose Foraminal": ["L4-L5"],
  "Contato com a Raiz Nervosa": ["L4", "L5"],
  "Raiz Nervosa Deslocada/Comprimida": ["None"],
  "Estenose do Recesso Lateral": ["None"]
  }

Example 2:

Input: {{report}}


"""

class SpineReportLLM:
    def __init__(
        self, llm: str = "gpt-4", api_key: str = os.environ["OPENAI_API_KEY"]
    ) -> None:
        """
        Args:
            llm (str): O modelo de linguagem a ser utilizado. Opções: "gemini" e "gpt-4".
            api_key (str): A chave de API necessária para acessar o modelo de linguagem.

        """
        self.api_key = api_key
        self.prompt_template = PromptTemplate(
            template=TEMPLATE, input_variables=["report"], template_format="jinja2"
        )
        self.output_parser = JsonOutputParser()
        self.model_temperature = 0.3
        self.model = self.set_model(llm)

    def set_model(self, model: str):
        """
        Define o modelo de linguagem a ser utilizado com base no parâmetro fornecido.

        Args:
            model (str): O modelo de linguagem a ser utilizado.

        Returns:
            BaseChatModel: O modelo de linguagem configurado.
        """
        if model == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=self.model_temperature,
                google_api_key=self.api_key,
            )
        return ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=self.model_temperature,
            openai_api_key=self.api_key,
        )

    def set_temperature(self, temperature: float = 0.3):
        """
        Define a temperatura a ser utilizada para a geração de texto.

        Args:
            temperature (float): A temperatura a ser utilizada.

        """
        self.model_temperature = temperature

    def analyze_report(self, report: str) -> dict:
        """
        Analisa um relatório utilizando o modelo de linguagem configurado.

        Args:
            report (str): O relatório a ser analisado.

        Returns:
            dict: O resultado da análise do relatório no formato json.

        """
        chain = self.prompt_template | self.model | self.output_parser
        return chain.invoke({"report": report})


if __name__ == "__main__":
    llm = SpineReportLLM("gpt-4")

    report = "RESSONÂNCIA MAGNÉTICA DA COLUNA LOMBOSSACRA \xa0 Método: \xa0 Realizadas sequências FSE ponderadas em T1 e T2. Planos de cortes múltiplos. \xa0 Análise: \xa0 Corpos vertebrais com alinhamento posterior, altura e sinal normais.  Discreta hipo-hidratação discal L4-L5 e  L5-S1.  Mínimo abaulamento discal posterior centrobilateral,  com extensão para as bases foraminais em L3-L4, sem compressão radicular.  Abaulamentos  discais  posteriores  centrobilaterais em L4-L5 e L5-S1 com extensão  para  as  bases  foraminais,   mantendo contato com emergência foraminal  da  raiz  de  L4  e  de  L5  à  esquerda, respectivamente, sem deslocá-las.  Demais discos intervertebrais com sinal normal, sem evidências de herniações relevantes. \xa0 Canal vertebral com amplitude normal. \xa0 Demais forames intervertebrais livres.  Leve artropatia degenerativa interapofisária L4-L5 e L5-S1.  Cone medular de contornos regulares e sinal homogêneo.  Edema na topografia dos ligamentos interespinhosos posterior de L5-S1, por sobrecarga mecânica. \xa0 Estruturas paravertebrais íntegras.  OPINIÃO:  Discreta discopatia degenerativa multissegmentar, com os achados acima descritos  Demais achados acima mencionados "

    result = llm.analyze_report(report)

    print(result)
    print(type(result))
