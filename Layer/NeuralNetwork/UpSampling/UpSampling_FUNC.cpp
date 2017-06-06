/*--------------------------------------------
 * FileName  : UpSampling_FUNC.cpp
 * LayerName : �A�b�v�T���v�����O
 * guid      : 14EEE4A7-1B26-4651-8EBF-B1156D62CE1B
 * 
 * Text      : �l���g�����A�����߂���
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"UpSampling_FUNC.hpp"


// {14EEE4A7-1B26-4651-8EBF-B1156D62CE1B}
static const Gravisbell::GUID g_guid(0x14eee4a7, 0x1b26, 0x4651, 0x8e, 0xbf, 0xb1, 0x15, 0x6d, 0x62, 0xce, 0x1b);

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
        L"�A�b�v�T���v�����O",
        L"�l���g�����A�����߂���"
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"UpScale",
            {
                L"�g����",
                L"",
            }
        },
        {
            L"PaddingType",
            {
                L"�p�f�B���O���",
                L"�p�f�B���O���s���ۂ̕��@�ݒ�",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PaddingType",
            {
                {
                    L"zero",
                    {
                        L"�[���p�f�B���O",
                        L"�s������0�Ŗ��߂�",
                    },
                },
                {
                    L"value",
                    {
                        L"�l",
                        L"�s�����Ɨאڂ���l���Q�Ƃ���",
                    },
                },
            }
        },
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
    };


    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =
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


    /** ItemData Learn <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Learn;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Learn;

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
	/** Name : �g����
	  * ID   : UpScale
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"UpScale",
			CurrentLanguage::g_lpItemData_LayerStructure[L"UpScale"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"UpScale"].text.c_str(),
			1, 1, 1,
			32, 32, 32,
			1, 1, 1));

	/** Name : �p�f�B���O���
	  * ID   : PaddingType
	  * Text : �p�f�B���O���s���ۂ̕��@�ݒ�
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PaddingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"zero",
			L"�[���p�f�B���O",
			L"�s������0�Ŗ��߂�");
		// 1
		pItemEnum->AddEnumItem(
			L"value",
			L"�l",
			L"�s�����Ɨאڂ���l���Q�Ƃ���");

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
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a learning setting.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSetting(void)
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

/** Create learning settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLearningSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


