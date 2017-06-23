#ifdef IODataLayer_image_EXPORTS
#define IODataLayer_image_API __declspec(dllexport)
#else
#define IODataLayer_image_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.Layer.IOData.IODataLayer_image.lib")
#endif
#endif

#include"../../../Layer/IOData/IIODataLayer_image.h"
#include"../../../Common/Guiddef.h"
#include"../../../Common/VersionCode.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

//======================================
// ���ʕ���
//======================================

/** ���C���[�̎��ʃR�[�h���擾����.
  * @param  o_layerCode		�i�[��o�b�t�@.
  * @return ���������ꍇ0. 
  */
extern IODataLayer_image_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode);

/** �o�[�W�����R�[�h���擾����.
  * @param  o_versionCode	�i�[��o�b�t�@.
  * @return ���������ꍇ0. 
  */
extern IODataLayer_image_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode);


/** ���C���[�w�K�ݒ���쐬���� */
extern IODataLayer_image_API SettingData::Standard::IData* CreateLearningSetting(void);
/** ���C���[�w�K�ݒ���쐬����
	@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
	@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
	@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
	@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
extern IODataLayer_image_API SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


//======================================
// CPU����
//======================================

/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerCPU(Gravisbell::U32 i_dataCount, U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);
/** ���͐M���f�[�^���C���[���쐬����.CPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);


//======================================
// GPU����
//======================================
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPU(Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);
/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param guid			���C���[��GUID.
	@param ioDataStruct	���o�̓f�[�^�\��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPUwithGUID(Gravisbell::GUID guid, Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);




}	// IOData
}	// Layer
}	// Gravisbell