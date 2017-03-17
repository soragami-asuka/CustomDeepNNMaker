
#include "stdafx.h"
#include "IODataLayer.h"
#include "Common/VersionCode.h"
#include "Library/SettingData/Standard/SettingData.h"

#include<vector>
#include<list>


namespace Gravisbell {
namespace Layer {
namespace IOData {
	
// {BEBA34EC-C30C-4565-9386-56088981D2D7}
static const GUID g_guid = { 0x6e99d406, 0xb931, 0x4de0, { 0xac, 0x3a, 0x48, 0xa3, 0x5e, 0x12, 0x98, 0x20 } };

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 


/** ���C���[�̎��ʃR�[�h���擾����.
  * @param  o_layerCode		�i�[��o�b�t�@.
  */
extern Gravisbell::ErrorCode GetLayerCode(GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** �o�[�W�����R�[�h���擾����.
  * @param  o_versionCode	�i�[��o�b�t�@.
  */
extern Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** ���C���[�w�K�ݒ���쐬���� */
extern SettingData::Standard::IData* CreateLearningSetting(void)
{
	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(g_guid, g_version);
	if(pLayerConfig == NULL)
		return NULL;

	return pLayerConfig;
}
/** ���C���[�w�K�ݒ���쐬����
	@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
	@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
	@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
	@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
extern SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
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


}	// IOData
}	// Layer
}	// Gravisbell