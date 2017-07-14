/*--------------------------------------------
 * FileName  : UpConvolution_FUNC.cpp
 * LayerName : �g����݂��݃j���[�����l�b�g���[�N
 * guid      : B87B2A75-7EA3-4960-9E9C-EAF43AB073B0
 * 
 * Text      : �t�B���^�ړ��ʂ�[Stride/UpScale]�Ɋg��������ݍ��݃j���[�����l�b�g���[�N.Stride=1,UpScale=2�Ƃ����ꍇ�A�����}�b�v�̃T�C�Y��2�{�ɂȂ�
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"UpConvolution_FUNC.hpp"


// {B87B2A75-7EA3-4960-9E9C-EAF43AB073B0}
static const Gravisbell::GUID g_guid(0xb87b2a75, 0x7ea3, 0x4960, 0x9e, 0x9c, 0xea, 0xf4, 0x3a, 0xb0, 0x73, 0xb0);

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
        L"�g����݂��݃j���[�����l�b�g���[�N",
        L"�t�B���^�ړ��ʂ�[Stride/UpScale]�Ɋg��������ݍ��݃j���[�����l�b�g���[�N.Stride=1,UpScale=2�Ƃ����ꍇ�A�����}�b�v�̃T�C�Y��2�{�ɂȂ�"
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"FilterSize",
            {
                L"�t�B���^�T�C�Y",
                L"��݂��݂��s�����͐M����",
            }
        },
        {
            L"Output_Channel",
            {
                L"�o�̓`�����l����",
                L"�o�͂����`�����l���̐�",
            }
        },
        {
            L"Stride",
            {
                L"�t�B���^�ړ���",
                L"��݂��݂��ƂɈړ�����t�B���^�̈ړ���",
            }
        },
        {
            L"UpScale",
            {
                L"�g����",
                L"",
            }
        },
        {
            L"Padding",
            {
                L"�p�f�B���O�T�C�Y",
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
	/** Name : �t�B���^�T�C�Y
	  * ID   : FilterSize
	  * Text : ��݂��݂��s�����͐M����
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : �o�̓`�����l����
	  * ID   : Output_Channel
	  * Text : �o�͂����`�����l���̐�
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Output_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : �t�B���^�ړ���
	  * ID   : Stride
	  * Text : ��݂��݂��ƂɈړ�����t�B���^�̈ړ���
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

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

	/** Name : �p�f�B���O�T�C�Y
	  * ID   : Padding
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Padding",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

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
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateRuntimeParameter();
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


