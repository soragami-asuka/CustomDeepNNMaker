/*--------------------------------------------
 * FileName  : Pooling_FUNC.cpp
 * LayerName : Pooling
 * guid      : EB80E0D0-9D5A-4ED1-A80D-A1667DE0C890
 * 
 * Text      : Pooling.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard/SettingData.h>

#include"Pooling_FUNC.hpp"


// {EB80E0D0-9D5A-4ED1-A80D-A1667DE0C890}
static const Gravisbell::GUID g_guid(0xeb80e0d0, 0x9d5a, 0x4ed1, 0xa8, 0x0d, 0xa1, 0x66, 0x7d, 0xe0, 0xc8, 0x90);

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
        L"Pooling",
        L"Pooling."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"FilterSize",
            {
                L"フィルタサイズ",
                L"Poolingを行う範囲",
            }
        },
        {
            L"Stride",
            {
                L"フィルタ移動量",
                L"畳みこみごとに移動するフィルタの移動量",
            }
        },
        {
            L"PoolingType",
            {
                L"Pooling種別",
                L"Poolingの方法設定",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PoolingType",
            {
                {
                    L"max",
                    {
                        L"MAXプーリング",
                        L"範囲内の最大値を使用する",
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
	/** Name : フィルタサイズ
	  * ID   : FilterSize
	  * Text : Poolingを行う範囲
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : フィルタ移動量
	  * ID   : Stride
	  * Text : 畳みこみごとに移動するフィルタの移動量
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : Pooling種別
	  * ID   : PoolingType
	  * Text : Poolingの方法設定
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PoolingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PoolingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PoolingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"max",
			L"MAXプーリング",
			L"範囲内の最大値を使用する");

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


