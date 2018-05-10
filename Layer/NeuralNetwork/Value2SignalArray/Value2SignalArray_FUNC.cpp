/*--------------------------------------------
 * FileName  : Value2SignalArray_FUNC.cpp
 * LayerName : 信号の配列から値へ変換
 * guid      : 6F6C75B8-9C41-43EA-8F80-98C6F1CF4A2D
 * 
 * Text      : 信号の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"Value2SignalArray_FUNC.hpp"


// {6F6C75B8-9C41-43EA-8F80-98C6F1CF4A2D}
static const Gravisbell::GUID g_guid(0x6f6c75b8, 0x9c41, 0x43ea, 0x8f, 0x80, 0x98, 0xc6, 0xf1, 0xcf, 0x4a, 0x2d);

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
        L"信号の配列から値へ変換",
        L"信号の配列から値へ変換する.\n最大値を取るCH番号を値に変換する.\n入力CH数＝分解能の整数倍である必要がある."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"inputMinValue",
            {
                L"入力最小値",
                L"",
            }
        },
        {
            L"inputMaxValue",
            {
                L"入力最大値",
                L"",
            }
        },
        {
            L"resolution",
            {
                L"分解能",
                L"",
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
	/** Name : 入力最小値
	  * ID   : inputMinValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"inputMinValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"inputMinValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"inputMinValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : 入力最大値
	  * ID   : inputMaxValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"inputMaxValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"inputMaxValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"inputMaxValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : 分解能
	  * ID   : resolution
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"resolution",
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].text.c_str(),
			2, 65535, 2));

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


