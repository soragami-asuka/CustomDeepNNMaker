/*--------------------------------------------
 * FileName  : ProbabilityArray2Value_FUNC.cpp
 * LayerName : �m���̔z�񂩂�l�֕ϊ�
 * guid      : 9E32D735-A29D-4636-A9CE-2C781BA7BE8E
 * 
 * Text      : �m���̔z�񂩂�l�֕ϊ�����.
 *           : �ő�l�����CH�ԍ���l�ɕϊ�����.
 *           : ����CH��������\�̐����{�ł���K�v������.
 *           : �w�K���̓��͂ɑ΂��鋳�t�M���͐���M���𒆐S�Ƃ������K���z�̕��ϒl���Ƃ�
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"ProbabilityArray2Value_FUNC.hpp"


// {9E32D735-A29D-4636-A9CE-2C781BA7BE8E}
static const Gravisbell::GUID g_guid(0x9e32d735, 0xa29d, 0x4636, 0xa9, 0xce, 0x2c, 0x78, 0x1b, 0xa7, 0xbe, 0x8e);

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
        L"�m���̔z�񂩂�l�֕ϊ�",
        L"�m���̔z�񂩂�l�֕ϊ�����.\n�ő�l�����CH�ԍ���l�ɕϊ�����.\n����CH��������\�̐����{�ł���K�v������.\n�w�K���̓��͂ɑ΂��鋳�t�M���͐���M���𒆐S�Ƃ������K���z�̕��ϒl���Ƃ�"
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"outputMinValue",
            {
                L"�o�͍ŏ��l",
                L"",
            }
        },
        {
            L"outputMaxValue",
            {
                L"�o�͍ő�l",
                L"",
            }
        },
        {
            L"resolution",
            {
                L"����\",
                L"",
            }
        },
        {
            L"variance",
            {
                L"���t�M���̕��U",
                L"",
            }
        },
        {
            L"allocationType",
            {
                L"���蓖�Ď��",
                L"CH�ԍ����l�ɕϊ����邽�߂̕ϊ����@",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"allocationType",
            {
                {
                    L"max",
                    {
                        L"�ő�l",
                        L"CH���̍ő�l���o�͂���",
                    },
                },
                {
                    L"average",
                    {
                        L"����",
                        L"CH�ԍ���CH�̒l���|�����킹���l�̕��ϒl���o�͂���(��������)",
                    },
                },
            }
        },
    };



    /** ItemData Runtiime Parameter <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Runtime = 
    {
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
	/** Name : �o�͍ŏ��l
	  * ID   : outputMinValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"outputMinValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMinValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMinValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : �o�͍ő�l
	  * ID   : outputMaxValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"outputMaxValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMaxValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMaxValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : ����\
	  * ID   : resolution
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"resolution",
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].text.c_str(),
			2, 65535, 2));

	/** Name : ���t�M���̕��U
	  * ID   : variance
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"variance",
			CurrentLanguage::g_lpItemData_LayerStructure[L"variance"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"variance"].text.c_str(),
			0.0000000000000000f, 65535.0000000000000000f, 1.0000000000000000f));

	/** Name : ���蓖�Ď��
	  * ID   : allocationType
	  * Text : CH�ԍ����l�ɕϊ����邽�߂̕ϊ����@
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"allocationType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"allocationType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"allocationType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"max",
			L"�ő�l",
			L"CH���̍ő�l���o�͂���");
		// 1
		pItemEnum->AddEnumItem(
			L"average",
			L"����",
			L"CH�ԍ���CH�̒l���|�����킹���l�̕��ϒl���o�͂���(��������)");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

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


