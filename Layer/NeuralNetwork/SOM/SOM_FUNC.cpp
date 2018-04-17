/*--------------------------------------------
 * FileName  : SOM_FUNC.cpp
 * LayerName : ���ȑg�D���}�b�v
 * guid      : AF36DF4D-9F50-46FF-A1C1-5311CA761F6A
 * 
 * Text      : ���ȑg�D���}�b�v.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"SOM_FUNC.hpp"


// {AF36DF4D-9F50-46FF-A1C1-5311CA761F6A}
static const Gravisbell::GUID g_guid(0xaf36df4d, 0x9f50, 0x46ff, 0xa1, 0xc1, 0x53, 0x11, 0xca, 0x76, 0x1f, 0x6a);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 



struct StringData
{
    std::wstring name;
    std::wstring text;
};

namespace DefaultLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = L"ja";

    /** Base */
    static const StringData g_baseData = 
    {
        L"���ȑg�D���}�b�v",
        L"���ȑg�D���}�b�v."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"InputBufferCount",
            {
                L"���̓o�b�t�@��",
                L"���C���[�ɑ΂�����̓o�b�t�@��",
            }
        },
        {
            L"DimensionCount",
            {
                L"������",
                L"���������}�b�v�̎�����",
            }
        },
        {
            L"ResolutionCount",
            {
                L"����\",
                L"�������Ƃ̕��𐫔\",
            }
        },
        {
            L"InitializeMinValue",
            {
                L"�������ŏ��l",
                L"�������Ɏg�p����l�̍ŏ��l",
            }
        },
        {
            L"InitializeMaxValue",
            {
                L"�������ő�l",
                L"�������Ɏg�p����l�̍ő�l",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
    };



    /** ItemData Runtiime Parameter <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Runtime = 
    {
        {
            L"SOM_L0",
            {
                L"�w�K�W��",
                L"�p�����[�^�X�V�̌W��",
            }
        },
        {
            L"SOM_ramda",
            {
                L"���Ԍ�����",
                L"�w�K�񐔂ɉ������w�K���̌�����.�l�������ق����������͒Ⴂ",
            }
        },
        {
            L"SOM_sigma",
            {
                L"����������",
                L"�X�V�̂�BMU�Ƃ̋����ɉ�����������.�l�������ق����������͒Ⴂ",
            }
        },
    };


    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Runtime =
    {
    };


}


namespace CurrentLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;


    /** Base */
    static StringData g_baseData = DefaultLanguage::g_baseData;


    /** ItemData Layer Structure <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;


    /** ItemData Runtime <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Runtime;

    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Runtime;

}



/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : ���̓o�b�t�@��
	  * ID   : InputBufferCount
	  * Text : ���C���[�ɑ΂�����̓o�b�t�@��
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"InputBufferCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"InputBufferCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"InputBufferCount"].text.c_str(),
			1, 65535, 200));

	/** Name : ������
	  * ID   : DimensionCount
	  * Text : ���������}�b�v�̎�����
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"DimensionCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"DimensionCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"DimensionCount"].text.c_str(),
			1, 16, 2));

	/** Name : ����\
	  * ID   : ResolutionCount
	  * Text : �������Ƃ̕��𐫔\
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"ResolutionCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"ResolutionCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"ResolutionCount"].text.c_str(),
			2, 65535, 10));

	/** Name : �������ŏ��l
	  * ID   : InitializeMinValue
	  * Text : �������Ɏg�p����l�̍ŏ��l
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"InitializeMinValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"InitializeMinValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"InitializeMinValue"].text.c_str(),
			-65535.0000000000000000f, 65535.0000000000000000f, 0.0000000000000000f));

	/** Name : �������ő�l
	  * ID   : InitializeMaxValue
	  * Text : �������Ɏg�p����l�̍ő�l
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"InitializeMaxValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"InitializeMaxValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"InitializeMaxValue"].text.c_str(),
			-65535.0000000000000000f, 65535.0000000000000000f, 1.0000000000000000f));

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a runtime parameters.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameter(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : �w�K�W��
	  * ID   : SOM_L0
	  * Text : �p�����[�^�X�V�̌W��
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"SOM_L0",
			CurrentLanguage::g_lpItemData_Learn[L"SOM_L0"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"SOM_L0"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.1000000014901161f));

	/** Name : ���Ԍ�����
	  * ID   : SOM_ramda
	  * Text : �w�K�񐔂ɉ������w�K���̌�����.�l�������ق����������͒Ⴂ
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"SOM_ramda",
			CurrentLanguage::g_lpItemData_Learn[L"SOM_ramda"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"SOM_ramda"].text.c_str(),
			0.0000000099999999f, 65535.0000000000000000f, 2500.0000000000000000f));

	/** Name : ����������
	  * ID   : SOM_sigma
	  * Text : �X�V�̂�BMU�Ƃ̋����ɉ�����������.�l�������ق����������͒Ⴂ
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"SOM_sigma",
			CurrentLanguage::g_lpItemData_Learn[L"SOM_sigma"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"SOM_sigma"].text.c_str(),
			0.0000000099999999f, 65535.0000000000000000f, 10.0000000000000000f));

	return pLayerConfig;
}

/** Create runtime parameter from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateRuntimeParameter();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


