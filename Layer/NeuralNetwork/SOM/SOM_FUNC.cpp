/*--------------------------------------------
 * FileName  : SOM_FUNC.cpp
 * LayerName : 自己組織化マップ
 * guid      : AF36DF4D-9F50-46FF-A1C1-5311CA761F6A
 * 
 * Text      : 自己組織化マップ.
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
        L"自己組織化マップ",
        L"自己組織化マップ."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"InputBufferCount",
            {
                L"入力バッファ数",
                L"レイヤーに対する入力バッファ数",
            }
        },
        {
            L"DimensionCount",
            {
                L"次元数",
                L"生成されるマップの次元数",
            }
        },
        {
            L"ResolutionCount",
            {
                L"分解能",
                L"次元ごとの分解性能",
            }
        },
        {
            L"InitializeMinValue",
            {
                L"初期化最小値",
                L"初期化に使用する値の最小値",
            }
        },
        {
            L"InitializeMaxValue",
            {
                L"初期化最大値",
                L"初期化に使用する値の最大値",
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
                L"学習係数",
                L"パラメータ更新の係数",
            }
        },
        {
            L"SOM_ramda",
            {
                L"時間減衰率",
                L"学習回数に応じた学習率の減衰率.値が高いほうが減衰率は低い",
            }
        },
        {
            L"SOM_sigma",
            {
                L"距離減衰率",
                L"更新個体とBMUとの距離に応じた減衰率.値が高いほうが減衰率は低い",
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
	/** Name : 入力バッファ数
	  * ID   : InputBufferCount
	  * Text : レイヤーに対する入力バッファ数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"InputBufferCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"InputBufferCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"InputBufferCount"].text.c_str(),
			1, 65535, 200));

	/** Name : 次元数
	  * ID   : DimensionCount
	  * Text : 生成されるマップの次元数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"DimensionCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"DimensionCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"DimensionCount"].text.c_str(),
			1, 16, 2));

	/** Name : 分解能
	  * ID   : ResolutionCount
	  * Text : 次元ごとの分解性能
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"ResolutionCount",
			CurrentLanguage::g_lpItemData_LayerStructure[L"ResolutionCount"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"ResolutionCount"].text.c_str(),
			2, 65535, 10));

	/** Name : 初期化最小値
	  * ID   : InitializeMinValue
	  * Text : 初期化に使用する値の最小値
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"InitializeMinValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"InitializeMinValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"InitializeMinValue"].text.c_str(),
			-65535.0000000000000000f, 65535.0000000000000000f, 0.0000000000000000f));

	/** Name : 初期化最大値
	  * ID   : InitializeMaxValue
	  * Text : 初期化に使用する値の最大値
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
	/** Name : 学習係数
	  * ID   : SOM_L0
	  * Text : パラメータ更新の係数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"SOM_L0",
			CurrentLanguage::g_lpItemData_Learn[L"SOM_L0"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"SOM_L0"].text.c_str(),
			0.0000000000000000f, 1.0000000000000000f, 0.1000000014901161f));

	/** Name : 時間減衰率
	  * ID   : SOM_ramda
	  * Text : 学習回数に応じた学習率の減衰率.値が高いほうが減衰率は低い
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"SOM_ramda",
			CurrentLanguage::g_lpItemData_Learn[L"SOM_ramda"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"SOM_ramda"].text.c_str(),
			0.0000000099999999f, 65535.0000000000000000f, 2500.0000000000000000f));

	/** Name : 距離減衰率
	  * ID   : SOM_sigma
	  * Text : 更新個体とBMUとの距離に応じた減衰率.値が高いほうが減衰率は低い
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


