//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __SEPARATEOUTPUT_DATA_BASE_H__
#define __SEPARATEOUTPUT_DATA_BASE_H__

#include<Layer/IO/ISingleInputLayerData.h>
#include<Layer/IO/IMultOutputLayerData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"SeparateOutput_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class SeparateOutput_LayerData_Base : public IO::ISingleInputLayerData, public IO::IMultOutputLayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[�f�[�^���ʗp��GUID */

		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct outputDataStruct;	/**< �o�̓f�[�^�\�� */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		SeparateOutput::LayerStructure layerStructure;	/**< ���C���[�\�� */


		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		SeparateOutput_LayerData_Base(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		virtual ~SeparateOutput_LayerData_Base();


		//===========================
		// ���ʏ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;

		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@param	i_config			�ݒ���
			@oaram	i_inputDataStruct	���̓f�[�^�\�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct);
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize );


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U32 GetUseBufferByteCount()const;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;

		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�̏o�͐惌�C���[��. */
		U32 GetOutputToLayerCount()const;

		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const;
		

		//===========================
		// �ŗL�֐�
		//===========================
	public:
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif