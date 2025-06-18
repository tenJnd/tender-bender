from src.agents.tender_llm_extractor import SemanticTenderDetail
from src.scrapers.nen_scraper import NenContractDetail

contract_metadata = NenContractDetail(
    procurement_procedure_name='Modul Cisco - OB7125-044',
    contracting_authority='Ministerstvo zahraničních věcí',
    nen_system_number='N006/25/V00015462', ien_system_number=None,
    contract_registration_number_in_the_vvz=None,
    current_status_of_the_procurement_procedure='Not terminated',
    division_into_lots='No',
    procurement_procedure_id_on_the_contracting_authoritys_profile='P25V00015462',
    public_contract_regime='Small-scale public contract',
    procurement_procedure_type='Otevřená výzva',
    specifications_of_the_procurement_procedure='otevřená výzva',
    type='Public supply contract', estimated_value_excl_vat=None, currency=None,
    date_of_publication_on_profile='05/15/2025, 10:40 AM',
    deadline_for_submitting_tenders='05/26/2025, 09:00 AM', name='Dagmar',
    surname='Bečková', email='dagmar.beckova@mzv.gov.cz', phone_1='+420 224183003',
    subject_matter_description='2x       modul\tCisco C3850-NM-2-10G',
    code_from_the_nipez_code_list='32423000-4',
    name_from_the_nipez_code_list='Síťové rozbočovače',
    main_place_of_performance='Hlavní město Praha',
    code_from_the_cpv_code_list='32423000-4',
    name_from_the_cpv_code_list='Síťové rozbočovače',
    subject_matter_name='Síťové rozbočovače',
    text_field_for_describing_the_place_of_performance='2x       modul\tCisco C3850-NM-2-10G',
    awarded_on_the_basis_of_a_framework_agreement='No', awarded_in_a_dns='No',
    the_result_of_the_pp_will_be_the_implementation_of_a_dns='No',
    this_is_a_framework_agreement='No', imported_public_contract='No',
    publication_records=[{
        'show_detail': 'Detail',
        'date_of_publication': '05/15/2025, 10:40 AM',
        'publications': 'Uveřejnění zadávacích podmínek',
        'date_of_withdrawal': '', 'withdrawn_by': '',
        'details_link': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462/detail-info/2865509816',
        'id': '2865509816'
    }],
    subject_matter_items=[],
    place_of_performance={'code': 'CZ010', 'place_of_performance': 'Hlavní město Praha'},
    detail_url='https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462',
    documents=[
        {
            'show_detail': 'Detail', 'file': 'VZ-044.docx',
            'document_type': 'Výzva k podání nabídky včetně zadávací dokumentace',
            'date_of_publication': '05/15/2025, 10:40 AM',
            'antivirus_scan': 'Dokument není zavirovaný',
            'details_link': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462/zadavaci-dokumentace/detail-dokument/2865511216',
            'id': '2865511216',
            'download_link': 'https://nen.nipez.cz/file?id=2865511216'
        }
    ],
    unmapped={'phone_2': '+420 734362655'}
)

semantic_tender_detail_dict = {'competition_risk_score': 6, 'data_completeness_score': 0.9, 'domain_tags': ['government'], 'estimated_value_eur_computed': None, 'extraction_confidence_score': 0.8, 'financial_attractiveness_score': 5, 'llm_extracted': {'budget_and_timeline_context': 'Zakázka musí být realizována do 14 dnů od podpisu smlouvy, což vyžaduje rychlou reakci a efektivní logistiku.', 'evaluation_criteria_summary': 'Nabídky budou hodnoceny podle nejnižší nabídkové ceny bez DPH.', 'executive_summary': 'Ministerstvo zahraničních věcí vyhlašuje veřejnou zakázku na dodávku dvou modulů Cisco C3850-NM-2-10G, které mohou být i repasované. Zakázka je zadávána mimo rámec zákona o zadávání veřejných zakázek a bude hodnocena na základě nejnižší nabídkové ceny. Místo plnění je Praha a doba plnění je 14 dní od podpisu smlouvy.', 'key_technologies_or_skills': ['Cisco'], 'main_challenges_or_risks': ['Ensuring timely delivery within 14 days'], 'matching_profile': {'attractiveness_justification': 'The financial attractiveness is moderate due to the small size of the tender and straightforward supply requirements.', 'collaboration_model': ['individual', 'small_team'], 'competition_intensity_score': 6, 'competition_justification': 'Moderate competition is expected as the product is standard and several vendors can supply it.', 'complexity_category': 'low', 'complexity_justification': 'The tender involves the supply of standard network modules with no complex installation or integration required.', 'financial_attractiveness_score': 5, 'preferred_company_size': ['small', 'medium'], 'required_experience_level': ['junior', 'mid'], 'technical_complexity_score': 3, 'tender_size_category': 'xs', 'urgency_score': 7}, 'scope_and_deliverables': 'Dodávka dvou modulů Cisco C3850-NM-2-10G, které mohou být i repasované, s místem plnění v Praze. Doba plnění je stanovena na 14 dní od podpisu smlouvy. Nabídky musí obsahovat celkovou cenu v CZK, která bude platná po celou dobu realizace zakázky.', 'searchable_keywords': ['Cisco', 'network module', 'public supply', 'Prague', 'government tender', 'Czech Republic'], 'semantic_tags': {'domain_expertise': ['government'], 'language_requirements': ['czech'], 'location_preferences': ['prague'], 'methodologies': [], 'required_certifications': [], 'service_types': ['supply'], 'technology_stack': ['Cisco C3850-NM-2-10G']}, 'target_vendor_profile': 'Ideální dodavatel je menší až středně velká firma s možností rychlé dodávky síťových modulů Cisco.', 'technical_summary': 'Předmětem zakázky je dodávka dvou modulů Cisco C3850-NM-2-10G. Nabídky musí být podány v českém jazyce a hodnoceny budou podle nejnižší nabídkové ceny bez DPH.'}, 'market_opportunity_score': 5, 'semantic_tags': ['Cisco', 'network module', 'public supply', 'Prague', 'government tender', 'Czech Republic'], 'service_tags': ['supply'], 'technical_complexity_score': 3, 'technology_tags': ['Cisco']}

